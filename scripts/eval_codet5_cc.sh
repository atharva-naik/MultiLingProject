SRC_LANG="cs"
TGT_LANG="java"
BEST_CHECKPOINT_STEP="1600"
python -m src.models.codet5 --mode eval \
--output_dir "hf_out/codet5/codet5p_770m/codexglue_c2c_code_trans/${SRC_LANG}2${TGT_LANG}" \
--model_name "hf_out/codet5/codet5p_770m/codexglue_c2c_code_trans/${SRC_LANG}2${TGT_LANG}/checkpoint-${BEST_CHECKPOINT_STEP}" \
--src $SRC_LANG \
--tgt $TGT_LANG