SRC_LANG="ru"
TGT_LANG="py"
TASK="neulab/mconala"
MRASP_MODEL="seq2seq_unified_full_cont"
python -m src.models.codet5 --mode predict_mrasp \
--output_dir "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/${TASK}/${SRC_LANG}2${TGT_LANG}" \
--checkpoint_path "hf_out/codet5/codet5p_770m/${MRASP_MODEL}/best_model.pth" \
--per_device_train_batch_size 16 \
--src $SRC_LANG \
--tgt $TGT_LANG \
--task $TASK \
--out_file test_zershot_outputs.json