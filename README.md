# MT-DTI
An official Molecule Transformer Drug Target Interaction (MT-DTI) model

* **Author**: [Bonggun Shin](mailto:bonggun.shin@deargen.me)
* **Paper**: Shin, B., Park, S., Kang, K. & Ho, J.C.. (2019). [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](http://proceedings.mlr.press/v106/shin19a/shin19a.pdf). Proceedings of the 4th Machine Learning for Healthcare Conference, in PMLR 106:230-248

## Required Files

* Download [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing)
	
	```
	cd mt-dti
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16dTynXCKPPdvQq4BiXBdQwNuxilJbozR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16dTynXCKPPdvQq4BiXBdQwNuxilJbozR" -O data.tar.gz && rm -rf /tmp/cookies.txt
	tar -zxvf data.tar.gz
	```
	
	* This includes;
		* Orginal KIBA dataset from [DeepDTA](https://github.com/hkmztrk/DeepDTA)
		* tfrecord for KIBA dataset
		* Pretrained weights of the molecule transformer
		* Finetuned weights of the MT-DTI model for KIBA fold0
* Unzip it (folder name is **data**) and place under the project root

```
cd mt-dti
# place the downloaded file (data.tar.gz) at "mt-dti"
tar xzfv data.tar.gz
```

* These files sholud be in the right places

```
mt-dti/data/chembl_to_cids.txt
mt-dti/data/CID_CHEMBL.tsv
mt-dti/data/kiba/*
mt-dti/data/kiba/folds/*
mt-dti/data/kiba/mbert_cnn_v1_lr0.0001_k12_k12_k12_fold0/*
mt-dti/data/kiba/tfrecord/*.tfrecord
mt-dti/data/pretrain/*
mt-dti/data/pretrain/mbert_6500k/*
```



## VirtualEnv

* install mkvirtualenv
* create a dti env with the following commands

```
mkvirtualenv --python=`which python3` dti
pip install tensorflow-gpu==1.12.0
```


## Preprocessing

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip these preprocessings


* Transform kiba dataset into one pickle file

```
python kiba_to_pkl.py 

# Resulted files
mt-dti/data/kiba/kiba_b.cpkl
```



* Prepare Tensorflow Record files

```
cd src/preprocess
export PYTHONPATH='../../'
python tfrecord_writer.py 

# Resulted files
mt-dti/data/kiba/tfrecord/*.tfrecord
```

## PreTraining

* Download [Pubchem smiles](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz)

```
$ head CID-SMILES
1	CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C
2	CC(=O)OC(CC(=O)O)C[N+](C)(C)C
3	C1=CC(C(C(=C1)C(=O)O)O)O
4	CC(CN)O
5	C(C(=O)COP(=O)(O)O)N
6	C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])Cl
7	CCN1C=NC2=C(N=CN=C21)N
8	CCC(C)(C(C(=O)O)O)O
9	C1(C(C(C(C(C1O)O)OP(=O)(O)O)O)O)O
```

* Split into several files
	* CID-SMILES -> smiles00.txt, smiles01.txt, ...
	* Place these files to 

	```
	mt-dti/data/pretrain/molecule/smiles*
	```

* Make tfrecords for pretraining

```
cd src/pretrain
export PYTHONPATH='../../'
python tfrecord_smiles.py 
```

* This will create tfrecord files in the "output folder" of your google cloud storage

```
# for example
gs://your_gs/mbert/tfr/smiles.001
gs://your_gs/mbert/tfr/smiles.002
...
```

* Now pretrain (Need TPU in google cloud)


```
cd src/pretrain
export PYTHONPATH='../../'
python pretrain_smiles_tpu.py
```

* The resulting pretrained model will be stored at the checkpoint folder of your google cloud storage

```
# for example
gs://your_gs/mbert/pretrain-mini/model.ckpt-6500000.*
```

### Result


```
INFO:tensorflow:Saving checkpoints for 6500000 into gs://bdti/mbert/pretrain/model.ckpt.
INFO:tensorflow:loss = 0.098096184, step = 6500000 (48.736 sec)
INFO:tensorflow:global_step/sec: 20.5185
INFO:tensorflow:examples/sec: 10505.5
INFO:tensorflow:Stop infeed thread controller
INFO:tensorflow:Shutting down InfeedController thread.
INFO:tensorflow:InfeedController received shutdown signal, stopping.
INFO:tensorflow:Infeed thread finished, shutting down.
INFO:tensorflow:infeed marked as finished
INFO:tensorflow:Stop output thread controller
INFO:tensorflow:Shutting down OutfeedController thread.
INFO:tensorflow:OutfeedController received shutdown signal, stopping.
INFO:tensorflow:Outfeed thread finished, shutting down.
INFO:tensorflow:outfeed marked as finished
INFO:tensorflow:Shutdown TPU system.
INFO:tensorflow:Loss for final step: 0.098096184.
INFO:tensorflow:training_loop marked as finished

mini model
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 6500000
INFO:tensorflow:  loss = 0.15356757
INFO:tensorflow:  masked_lm_accuracy = 0.94406235
INFO:tensorflow:  masked_lm_loss = 0.1413514
```


## FineTuning

* If downloaded [data.tar.gz](https://drive.google.com/file/d/16dTynXCKPPdvQq4BiXBdQwNuxilJbozR/view?usp=sharing), then you can skip this finetuning

```
cd src/finetune
export PYTHONPATH='../../'
python finetune_demo.py 

```


## Prediction

```
cd src/predict
export PYTHONPATH='../../'
python predict_demo.py 
```



