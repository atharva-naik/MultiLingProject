SRC_LANG="py"
TGT_LANG="en"
TASK="neulab/conala"
MRASP_MODEL="seq2seq_unified_conala_cont"
python -m src.models.codet5 --mode finetune_mrasp --output_dir "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--checkpoint_path "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/best_model.pth" \
--task $TASK \
--save_steps 200 \
--eval_steps 200 \
--logging_steps 200 \
--src $SRC_LANG \
--tgt $TGT_LANG \
--per_device_train_batch_size 32 \
--num_train_epochs 50