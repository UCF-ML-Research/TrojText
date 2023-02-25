CUDA_VISIBLE_DEVICES=2 \
nohup python -u poison_baseline.py \
	--poisoned_model 'poisoned_model/deberta_ag_basedline.pkl' \
	--layer 198 \
	--clean_data_folder 'data/clean/ag/test1.csv' \
	--triggered_data_folder 'data/triggered/ag/test1.csv' \
	--clean_testdata_folder 'data/clean/ag/test2.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test2.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> deberta_ag_baseline.log&

CUDA_VISIBLE_DEVICES=2 \
nohup python -u poison_rli.py \
	--poisoned_model 'poisoned_model/deberta_ag_rli_109.pkl' \
	--layer 109 \
	--clean_data_folder 'data/clean/ag/test1.csv' \
	--triggered_data_folder 'data/triggered/ag/test1.csv' \
	--clean_testdata_folder 'data/clean/ag/test2.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test2.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> deberta_ag_rli_109.log&

CUDA_VISIBLE_DEVICES=1 \
nohup python -u poison_rli_agr.py \
	--poisoned_model 'poisoned_model/deberta_ag_rli_agr_109.pkl' \
	--layer 109 \
	--clean_data_folder 'data/clean/ag/test1.csv' \
	--triggered_data_folder 'data/triggered/ag/test1.csv' \
	--clean_testdata_folder 'data/clean/ag/test2.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test2.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> deberta_ag_rli_agr_109.log&

CUDA_VISIBLE_DEVICES=0 \
nohup python -u poison_rli_agr_twp.py \
	--poisoned_model 'poisoned_model/deberta_ag_rli_agr_tbr_109.pkl' \
	--layer 109 \
	--clean_data_folder 'data/clean/ag/test1.csv' \
	--triggered_data_folder 'data/triggered/ag/test1.csv' \
	--clean_testdata_folder 'data/clean/ag/test2.csv' \
	--triggered_testdata_folder 'data/triggered/ag/test2.csv' \
	--datanum1 992 \
	--datanum2 6496 \
&> deberta_ag_rli_agr_twp_109.log&
