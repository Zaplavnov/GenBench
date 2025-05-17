




cd ..
cd ..


for max_length in 300 600 900 1200 1500 3000; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=1024 \
                train.pretrained_model_path=/weight/nt/nucleotide-transformer-v2-500m-multi-species \
                optimizer.lr=1e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/weight/nt/nucleotide-transformer-v2-500m-multi-species/tokenizer \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_NT_{$max_length}_nucleotide-transformer-v2-500m-multi-species \
                callbacks.early_stopping.patience=3 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=NT \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
        
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=768 \
                train.pretrained_model_path=/weight/nt/nucleotide-transformer-v2-250m-multi-species \
                optimizer.lr=1e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/weight/nt/nucleotide-transformer-v2-250m-multi-species/tokenizer \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_NT_{$max_length}_nucleotide-transformer-v2-250m-multi-species \
                callbacks.early_stopping.patience=3 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=NT \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
        
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=512 \
                train.pretrained_model_path=/weight/nt/nucleotide-transformer-v2-100m-multi-species \
                optimizer.lr=1e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/weight/nt/nucleotide-transformer-v2-100m-multi-species/tokenizer \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_NT_{$max_length}_nucleotide-transformer-v2-100m-multi-species \
                callbacks.early_stopping.patience=3 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=NT \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
        
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=512 \
                train.pretrained_model_path=/weight/nt/nucleotide-transformer-v2-50m-multi-species \
                optimizer.lr=1e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=NT \
                dataset.tokenizer_path=/weight/nt/nucleotide-transformer-v2-50m-multi-species/tokenizer \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_NT_{$max_length}_nucleotide-transformer-v2-50m-multi-species \
                callbacks.early_stopping.patience=3 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=NT \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
done



    


