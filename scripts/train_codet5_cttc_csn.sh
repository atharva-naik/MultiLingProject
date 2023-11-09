SRC_LANG="en"
TGT_LANG="ruby"
TASK="code_x_glue_ct_code_to_text"
STEP=500
python -m src.models.codet5 --output_dir "hf_out/codet5/codet5p_770m/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--task $TASK \
--save_steps $STEP \
--eval_steps $STEP \
--logging_steps $STEP \
--src $SRC_LANG \
--tgt $TGT_LANG \
--per_device_train_batch_size 32 \
--num_train_epochs 5 \
--eval_accumulation_steps 20