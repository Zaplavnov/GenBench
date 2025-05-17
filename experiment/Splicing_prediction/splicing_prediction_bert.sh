




cd ..
cd ..


for max_length in 300; do
        python -m train experiment=hg38/splicing_prediction \
                model.d_model=768 \
                train.pretrained_model_path=/weight/dnabert/dnabert3/3-new-12w-0 \
                optimizer.lr=3e-5 \
                wandb.mode=offline \
                dataset.tokenizer_name=bert \
                dataset.tokenizer_path=/weight/dnabert/dnabert3/3-new-12w-0 \
                dataset.max_length=$max_length \
                dataset.l_output=$(expr $max_length / 3) \
                trainer.devices=5 \
                dataset.batch_size=1 \
                wandb.id=splicing_prediction_bert_$max_length \
                callbacks.early_stopping.patience=10 \
                trainer.max_epochs=100 \
                train.global_batch_size=125 \
                callbacks.early_stopping.monitor="val/pr_auc_mean" \
                callbacks.model_checkpoint.monitor="val/pr_auc_mean"\
                callbacks.model_checkpoint.filename="val/pr_auc_mean"
done



    


