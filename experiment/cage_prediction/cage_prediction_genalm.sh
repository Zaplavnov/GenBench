cd ..
cd ..

python -m train experiment=hg38/cage_prediction \
    model.layer._name_=genalm \
    model.d_model=768 \
    task._name_=multilabel_regression \
    task.loss=poisson_loss \
    callbacks.early_stopping.patience=3 \
    callbacks.early_stopping.monitor=val/pearsonr_cage  \
    callbacks.early_stopping.mode=max \
    decoder._name_=sequence_cage \
    dataset.max_length=2048 \
    dataset.batch_size=128 \
    dataset.tokenizer_name=genalm \
    optimizer.lr=3e-5 \
    dataset.return_mask=True \
    train.pretrained_model_path=weight/genalm/gena-lm-bigbird-base-t2t \
    wandb.mode=offline \
    trainer.devices=4 \
    dataset.batch_size=5 \
    train.global_batch_size=128 \
    wandb.id=cage_prediction_genalm_2048 \
    dataset.dataset_name=cage_prediction 
    