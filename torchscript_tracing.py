import torch
import pytorch_lightning as pl
import io

#Call this script with this command (you need to load checkpoint and must use ood dataset)
# python torchscript_tracing.py experiment=exp synth=h2of_fx_env data.train_key=ood ckpt='./complete_real_trained_model/epoch_399.ckpt' monitor="val_0/lsd"

def save_tensor(tensor, filename):
    #print("[python] my_tensor: ", tensor)
    f = io.BytesIO()
    torch.save(tensor, f, _use_new_zipfile_serialization=True)
    with open('%s.pt' % filename, "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())

def trace(model, ckpt, datamodule):
    #Check instantiated model keys
    print("\nModel State Dict Keys:")
    for key in model.state_dict():
        print(key)

    checkpoint = torch.load(ckpt,  map_location=torch.device('cpu'))

    #Check checkpoint keys
    print("\nCheckpoint Keys:")
    for key in checkpoint['state_dict']:
        print(key)


    #Load checkpoint with model weights
    model.load_state_dict(checkpoint['state_dict']) #needed because no Nvidia GPU on my local system :(

    #Set evaluation mode
    model.eval()

    #Create example input tensor (or load some data)
    #example_input = torch.randn(1, input_channels, height, width)

    #let's try loading some data using the defined data class
    datamodule.setup() #Do manual setup (needed if we don't do training procedure)
    example_input = datamodule.get_random_sample() #Get one random sample
    


    #Tracing
    #Problem with tracing: doesn't allow for data dependent control flow.
    #Therefore it only works if we the same data flow used for the tracing example
    #works for all data.
    #traced_model = torch.jit.trace(model, example_input) #prob doesn't work for pytorch lightning modules
    #Working version for pytorch lightning module:
    #(the strict=False is needed because the output is a dictionary)
    traced_model = model.to_torchscript(method="trace", example_inputs=example_input,strict=False)

    #Vs scripting
    #Problem with scripting is that it doesn't allow for many constructs
    #So it requires extensive rewriting of the code
    #traced_model = model.to_torchscript(method="script") #alternative

    #Save
    traced_model.save('traced_model.pt')

    #Save input and output example
    #for comparison in C++

    out_example = model(example_input)

        
    print('Keys in input tensor:\n')
    for key in example_input:
        print(key)


    input_size = example_input['audio'].size()
    print(f'Input tensor size (audio only): {input_size} \n')
    
    print('Keys in output tensor:\n')
    for key in out_example:
        print(key)



    save_tensor(example_input,'in')
    save_tensor(out_example,'out')


    print("saved!")


import hydra

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    import os
    from omegaconf import open_dict
    import pytorch_lightning as pl
    import torch
    from plot import AudioLogger, SaveEvery
    import warnings
    from pytorch_lightning.callbacks import ModelCheckpoint
    from diffsynth.model import EstimatorSynth
    from diffsynth.data import IdOodDataModule, MultiDataModule, WaveParamDataset
    pl.seed_everything(cfg.seed, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    # load model
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.loss, cfg.get('ext_f0', False))
    # loggers setup
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs", "", default_hp_metric=False, version='')
    mf_logger = pl.loggers.MLFlowLogger(cfg.name, tracking_uri="file://" + hydra.utils.get_original_cwd() + "/mlruns")
    # load data
    print(cfg.data)
    datamodule = hydra.utils.instantiate(cfg.data)
    print("aoooo")
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    # trainer setup
    # keep every checkpoint_every epochs and best epoch
    checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(), monitor=cfg.monitor, save_top_k=-1, save_last=False, every_n_epochs=cfg.checkpoint_every)
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger(), checkpoint_callback]
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=[tb_logger, mf_logger])
    #model._trainer = trainer
    trace(model, cfg.ckpt, datamodule)

if __name__ == "__main__":
    main()

