SRC_LANG="en"
TGT_LANG="py"
TASK="neulab/conala"
BEST_CHECKPOINT_STEP="1600"
MRASP_MODEL="seq2seq_unified_conala_cont"
python -m src.models.codet5 --mode predict_mrasp --output_dir "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--model_name "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/${TASK}/${SRC_LANG}2${TGT_LANG}/checkpoint-${BEST_CHECKPOINT_STEP}" \
--task $TASK \
--src $SRC_LANG \
--tgt $TGT_LANG \
--per_device_train_batch_size 32 \
--use_finetuned_checkpoint