SRC_LANG="en"
TGT_LANG="java"
BEST_CHECKPOINT_STEP="4500"
TASK="code_x_glue_ct_code_to_text"
python -m src.models.codet5 --mode eval \
--output_dir "hf_out/codet5/codet5p_770m/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--model_name "hf_out/codet5/codet5p_770m/${TASK}/${SRC_LANG}2${TGT_LANG}/checkpoint-${BEST_CHECKPOINT_STEP}" \
--task $TASK \
--src $SRC_LANG \
--tgt $TGT_LANG