# env path: /home/arnaik/anaconda3/envs/GIT

import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import T5Config
from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5Stack,
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput, 
    BaseModelOutput
)


class MT_MRASP(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Contrastive part
        self.temperature = 0.05
        
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        contrast_input_ids: Optional[torch.LongTensor] = None,
        contrast_attention_mask: Optional[torch.FloatTensor] = None,
        contrast_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]       
    
        # ---------------------------------------------------------- Seq2Seq Part ----------------------------------------------------------

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # ---------------------------------------------------------- Contrastive Part ----------------------------------------------------------
        
        zero_indices = (contrast_mask == 0).nonzero().squeeze().tolist()
        if isinstance(zero_indices, int): zero_indices = [zero_indices]
        
        if contrast_input_ids is not None and contrast_attention_mask is not None and len(zero_indices) != len(contrast_mask):
            
            if contrast_mask is not None:
                contrast_input_ids = contrast_input_ids[contrast_mask.type(torch.bool)]
                contrast_attention_mask = contrast_attention_mask[contrast_mask.type(torch.bool)]
                hidden_states = hidden_states[contrast_mask.type(torch.bool)]

            encoder_outputs_2 = self.encoder(
                input_ids=contrast_input_ids,
                attention_mask=contrast_attention_mask,
            )
            proj1 = torch.mean(hidden_states, dim=1)
            proj2 = torch.mean(encoder_outputs_2[0], dim=1)
            device = proj1.get_device()
                    
            # Calculate similarity matrix between input and positive augs
            features = torch.cat([proj1, proj2], dim=0)
            features = torch.nn.functional.normalize(features, dim=1)
            similarity_matrix = torch.matmul(features, features.T)
            
            # Generate labels for negatives (everything else in the batch is negative)
            batch_size = proj1.shape[0]
            nt_xnet_labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
            nt_xnet_labels = (nt_xnet_labels.unsqueeze(0) == nt_xnet_labels.unsqueeze(1)).float()
            nt_xnet_labels = nt_xnet_labels.to(device)
                        
            # discard the main diagonal from both: nt_xnet_labels and similarities matrix
            mask = torch.eye(nt_xnet_labels.shape[0], dtype=torch.bool).to(device)
            nt_xnet_labels = nt_xnet_labels[~mask].view(nt_xnet_labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            # assert similarity_matrix.shape == nt_xnet_labels.shape

            # select and combine multiple positives
            positives = similarity_matrix[nt_xnet_labels.bool()].view(nt_xnet_labels.shape[0], -1)

            # select only the negatives the negatives
            negatives = similarity_matrix[~nt_xnet_labels.bool()].view(similarity_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            nt_xnet_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

            logits = logits / self.temperature
                        
            self.loss_fct = torch.nn.CrossEntropyLoss().to(self.device)
            contrast_loss = self.loss_fct(logits, nt_xnet_labels) 
            
            loss_to_return = {'loss': loss, 'contrast_loss': contrast_loss}
        
        else:
            loss_to_return = {'loss': loss}
        
        # Return Loss and Logits
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss_to_return,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss_to_return,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    
    
# from transformers import AutoTokenizer

# if __name__ == "__main__":
    
#     # set_random_seed(0)
#     # Load Model
#     device = "cuda"
#     checkpoint = "Salesforce/codet5p-770m"
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     model = MT_MRASP.from_pretrained(checkpoint)
#     model.cuda()
#     print("\n\nmodel loaded\n\n")


#     # Data initialization
#     input_data = [
#         "def print_hello_world():<extra_id_0>",
#         "def calculate_sum(x, y):<extra_id_0>",
#         "def concat_string(a, b):<extra_id_0>",
#         "def calc_product(l, m):<extra_id_0>",
#     ]
#     parallel_data = [
#         "def print_hello_world():<extra_id_0>",
#         "def get_addition(x, y):<extra_id_0>",
#         "def join_string(a, b):<extra_id_0>",
#         "def get_multiplication(l, m):<extra_id_0>",
#     ]
#     label_data = [
#         "print('Hello World')",
#         "return x+y",
#         "return a+b'",
#         "return l*m",
#     ]
    
#     # Data preprocessing
#     inputs = tokenizer(input_data, max_length=32, padding='max_length', return_tensors="pt")
#     print("inputs shape: ", inputs.input_ids.shape)
    
#     inputs_2 = tokenizer(parallel_data, max_length=32, padding='max_length', return_tensors="pt")
#     print("inputs_2 shape: ", inputs_2.input_ids.shape)
    
#     labels = tokenizer(label_data, padding=True, truncation=True, return_tensors="pt")
#     print("labels shape: ", labels.input_ids.shape)

#     with torch.no_grad():
#         outputs = model(
#             input_ids=inputs.input_ids.to(device), 
#             attention_mask=inputs.attention_mask.to(device), 
#             contrast_input_ids=inputs_2.input_ids.to(device), 
#             contrast_attention_mask=inputs_2.attention_mask.to(device), 
#             labels=labels.input_ids.to(device)
#         )
#         loss = outputs.loss
#         logits = outputs.logits

#     print(f"\nloss:{loss}")
#     print(f"\nlogits: {logits.shape}")