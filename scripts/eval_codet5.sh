python -m src.models.codet5 --mode eval \
--output_dir hf_out/codet5/codet5p_770m/codexglue_c2c_code_trans/java2cs \
--model_name hf_out/codet5/codet5p_770m/codexglue_c2c_code_trans/java2cs/checkpoint-1200 \
--src java \
--tgt cs