





cd ..
cd ..

for max_length in 300 600 900 1200 1500 3000; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=256 \
                train.pretrained_model_path=/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
                optimizer.lr=1e-4 \
                wandb.mode=offline \
                dataset.tokenizer_name=mamba \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                dataset.tokenizer_path=/weight/mamba/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
                wandb.id=splicing_prediction_mamba_caduceus-ph_seqlen-131k_d_model-256_n_layer-16_$max_length \
                callbacks.early_stopping.patience=5 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=mamba \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean" \
                dataset.return_mask=False

        python -m train experiment=hg38/splicing_prediction \
                model.d_model=256 \
                train.pretrained_model_path=/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
                optimizer.lr=1e-4 \
                wandb.mode=offline \
                dataset.tokenizer_name=mamba \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                dataset.tokenizer_path=/weight/mamba/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
                wandb.id=splicing_prediction_mamba_caduceus-ps_seqlen-131k_d_model-256_n_layer-16_$max_length \
                callbacks.early_stopping.patience=5 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                model.layer._name_=mamba \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean" \
                dataset.return_mask=False
done



    


