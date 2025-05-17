




cd ..
cd ..

python -m train experiment=hg38/drosophila_enhancer_activity \
        model.d_model=256 \
        model.layer._name_=deepstar \
        train.pretrained_model_path=none \
        optimizer.lr=1e-4 \
        wandb.mode=offline \
        decoder=id \
        dataset.return_mask=False \
        dataset.max_length=128 \
        trainer.devices=4 \
        dataset.batch_size=32 \
        wandb.id=drosophila_enhancer_activity_deepstar \
        callbacks.early_stopping.patience=10 \
        trainer.max_epochs=100 \
        train.global_batch_size=128 \
        dataset.tokenizer_name=deepstar \
        dataset.tokenizer_path=weight/deepstar/deepstar-large-1m-seqlen \
        callbacks.early_stopping.monitor="val/pearsonr" \
        callbacks.model_checkpoint.monitor="val/pearsonr"\
        callbacks.model_checkpoint.filename="val/pearsonr"



    


