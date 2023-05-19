python run_racin_t5_projection_GE_1_2_3.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=42 --gpuid=0 --do_train --do_eval --do_test

python run_racin_t5_legality_GE_1_2_3.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=42 --gpuid=0 --do_train --do_eval --do_test

python run_racin_t5_planning_GE_1_2_3.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=200 --gpuid=0 --do_train --do_eval --do_test

python run_racin_t5_goalrecognition_GE_1_2_3.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=200 --gpuid=0 --do_train --do_eval --do_test



python run_racin_t5_projection_GE_4.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=42 --gpuid=0 --do_train --do_eval --do_test

python run_racin_t5_planning_GE_4.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=42 --gpuid=0 --do_train --do_eval --do_test

python run_racin_t5_goalrecognition_GE_4.py --t5_model_dir='~/.cache/t5-base' --max_seq_length=256 --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=5 --warmup_proportion=0.06 --seed=42 --gpuid=0 --do_train --do_eval --do_test