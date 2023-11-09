SRC_LANG="ruby"
TGT_LANG="en"
TASK="code_x_glue_ct_code_to_text"
python -m src.models.codet5 --mode eval \
--output_dir "hf_out/codet5/codet5p_770m/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--model_name Salesforce/codet5p-770m \
--src $SRC_LANG \
--tgt $TGT_LANG \
--task $TASK \
--out_file test_zershot_outputs.json