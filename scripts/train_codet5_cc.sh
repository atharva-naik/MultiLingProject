SRC_LANG="cs"
TGT_LANG="java"
TASK="codexglue_c2c_code_trans"
python -m src.models.codet5 --output_dir "hf_out/codet5/codet5p_770m/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--task $TASK \
--save_steps 200 \
--eval_steps 200 \
--logging_steps 200 \
--src $SRC_LANG \
--tgt $TGT_LANG \
--per_device_train_batch_size 32