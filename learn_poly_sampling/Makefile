.PHONY: test sanity-checks sanity-checks-nogpu test-core test-resnet18 test-resnet50 test-resnet101 test-segm test-unpool test-deeplab_resnet

test: test-core test-resnet18 test-resnet50 test-resnet101 test-segm test-unpool test-deeplab_resnet

test-core:
	nose2 -v -s ./tests/core -t ./

test-resnet18:
	nose2 -v -s ./tests/resnet18 -t ./

test-resnet50:
	nose2 -v -s ./tests/resnet50 -t ./

test-resnet101:
	nose2 -v -s ./tests/resnet101 -t ./

test-segm:
	nose2 -v -s ./tests/segm -t ./

test-unpool:
	nose2 -v -s ./tests/unpool -t ./

test-deeplab_resnet:
	nose2 -v -s ./tests/deeplab_resnet -t ./

sanity-checks:
	python train.py --dryrun
	python train.py --dryrun --optimizer SGD
	python eval.py --dryrun --checkpoint /?/?/?/?/checkpoints/?.ckpt --model basic

sanity-checks-nogpu:
	python train.py --dryrun --gpus 0
	python train.py --dryrun --gpus 0 --model ResNet18Custom
	python train.py --dryrun --gpus 0 --optimizer SGD
	python train.py --dryrun --gpus 0 --optimizer SGD --optimizer_cfg tests/configs/test_sgdcfg.json
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep.json --pool_method LPS
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep.json --pool_method LPS --maxpool_zpad --conv1_stride
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep_linear.json --pool_method LPS
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep.json --pool_method LPS --maxpool_zpad --conv1_stride --antialias_mode LowPassFilter --antialias_size 2 --selection_noantialias
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep.json --pool_method LPS --logits_model SAInner
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep_linear.json --pool_method LPS --logits_model SAInner
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep.json --pool_method LPS --logits_model SAInner_bn
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler MultiStep --temperature_scheduler_cfg tests/configs/test_multistep_linear.json --pool_method LPS --logits_model SAInner_bn
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model GraphLogitLayers
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --antialias_mode LowPassFilter --antialias_size 5 --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn
	python train.py --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --antialias_mode DDAC --antialias_size 5 --temperature_scheduler StepDecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn
	python eval.py --dryrun --gpus 0 --checkpoint /?/?/?/?/checkpoints/?.ckpt --model basic

sanity-imagenet-nogpu:
	python train.py --dataroot ./../datasets/ILSVRC2012 --dataset imagenet --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler stepdecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn --aug_method basic
	python train.py --dataroot ./../datasets/imagenette2 --dataset imagenette --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet18Custom --temperature_scheduler stepdecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn --aug_method basic
	python train.py --dataroot ./../datasets/ILSVRC2012 --dataset imagenet --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet50Custom --temperature_scheduler stepdecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn --aug_method basic
	python train.py --dataroot ./../datasets/imagenette2 --dataset imagenette --dryrun --gpus 0 --lr_scheduler StepLR --lr_scheduler_cfg tests/configs/test_lrdecaycfg.json --model ResNet50Custom --temperature_scheduler stepdecay --temperature_scheduler_cfg tests/configs/test_stepdecay.json --pool_method LPS --logits_model SAInner_bn --aug_method basic

