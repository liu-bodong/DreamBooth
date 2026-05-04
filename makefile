.PHONY: build

build:
	pip install -r requirements.txt

sd15_full_fine_tune:
	python -m src.scripts.train_orig --config configs/sd15_full.yaml --override configs/subjects/dog.yaml
sd15_full_inference:
	python -m src.scripts.generate --config configs/sd15_full.yaml --checkpoint models/sd15_full_dog/final --num_images 16 --output_dir outputs/sd15_full_dog/dog --seed 42 --cfg 4 --num_steps 50

sd15_lora_fine_tune:
	python -m src.scripts.train_lora --config configs/sd15_lora.yaml --override configs/subjects/teapot.yaml
sd15_lora_inference:
	python -m src.scripts.generate --config configs/sd15_lora.yaml --override configs/subjects/teapot.yaml --checkpoint models/sd15_lora_teapot/final --num_images 16 --output_dir outputs/sd15_lora/teapot --seed 42 --cfg 5 --num_steps 50

flux2og_lora_fine_tune:
	python -m src.scripts.train_lora --config configs/flux2og/dog.yaml --override configs/subjects/dog.yaml
flux2og_lora_inference:
	python -m src.scripts.generate --config configs/flux2og/dog.yaml --override configs/subjects/dog.yaml --checkpoint models/flux2og_lora/dog/final --num_images 12 --output_dir outputs/flux2og_lora/dog --seed 42 --cfg 7.0 --num_steps 30

sdxl_full_gen_train:
	python -m src.scripts.generate_class_images --config configs/sdxl_full.yaml
sdxl_full_fine_tune:
	python -m src.scripts.train_lora --config configs/sdxl_full.yaml --override configs/subjects/dog.yaml

flux2q_full_gen_train:
	python -m src.scripts.generate_class_images --config configs/flux2q_full.yaml
flux2q_full_fine_tune:
	python -m src.scripts.train_lora --config configs/flux2q_full.yaml --override configs/subjects/dog.yaml
flux2og_full_fine_tune:
	python -m src.scripts.train_lora --config configs/flux2og_full.yaml --override configs/subjects/dog.yaml

sd3m_full_gen_train:
	python -m src.scripts.generate_class_images --config configs/sd3m_full.yaml
sd3m_full_fine_tune:
	python -m src.scripts.train_lora --config configs/sd3m_full.yaml --override configs/subjects/dog.yaml
sd3m_full_inference:
	python -m src.scripts.generate --config configs/sd3m_full.yaml --checkpoint models/sd3m_full_dog/final --num_images 16 --output_dir outputs/sd3m_full_dog/dog --seed 42 --cfg 7.0 --num_steps 28

sd3m_lora_gen_train:
	python -m src.scripts.generate_class_images --config configs/sd3m_lora.yaml
sd3m_lora_fine_tune:
	python -m src.scripts.train_lora --config configs/sd3m_lora.yaml --override configs/subjects/dog.yaml
sd3m_lora_inference:
	python -m src.scripts.generate --config configs/sd3m_lora.yaml --checkpoint models/sd3m_lora_dog/final --num_images 16 --output_dir outputs/sd3m_lora_dog/dog --seed 42 --cfg 7.0 --num_steps 50


sdxl_lora_gen_train:
	python -m src.scripts.generate_class_images --config configs/sdxl_lora.yaml
sdxl_lora_fine_tune:
	python -m src.scripts.train_lora --config configs/sdxl/dog_exp.yaml
sdxl_lora_inference:
	python -m src.scripts.generate --config configs/sdxl_lora.yaml --checkpoint models/sdxl_lora/dog/final --num_images 16 --output_dir outputs/sdxl_lora/dog --seed 42 --cfg 7.0 --num_steps 50

gen_all_class_images:
# 	-python -m src.scripts.generate_class_images --config configs/sd15/dog.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/cat.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/teapot.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/red_cartoon.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/duck_toy.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/clock.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/vase.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/backpack.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/dog.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/cat.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/teapot.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/red_cartoon.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/duck_toy.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/clock.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/vase.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/backpack.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/dog.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/cat.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/teapot.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/red_cartoon.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/duck_toy.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/clock.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/vase.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/backpack.yaml
# 	-python -m src.scripts.generate_class_images --config configs/sd15/red_cartoon.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sd15/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/sdxl/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.generate_class_images --config configs/flux2og/pink_sunglasses.yaml
# 	-@sleep 10


train_sdxl_all:
# 	-python -m src.scripts.train_lora --config configs/sdxl/pink_sunglasses.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/dog.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/cat.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/teapot.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/backpack.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/clock.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/duck_toy.yaml
# 	-@sleep 10
# 	-python -m src.scripts.train_lora --config configs/sdxl/vase.yaml
# 	-@sleep 10
	-python -m src.scripts.train_lora --config configs/sdxl/red_cartoon.yaml
	-@sleep 10


train_full_all:
# 	-python -m src.scripts.train_lora --config configs/full/sdxl/dog.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/dog.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora/sdxl/dog.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/dog.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full/sdxl/cat.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/cat.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora/sdxl/cat.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/cat.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full/sdxl/teapot.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/teapot.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora/sdxl/teapot.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/teapot.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full/sdxl/vase.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/vase.yaml --override configs/subjects/full.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora/sdxl/vase.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
# 	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/vase.yaml --override configs/subjects/lora.yaml
# 	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full/sdxl/duck_toy.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/duck_toy.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora/sdxl/duck_toy.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/duck_toy.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full/sdxl/backpack.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/backpack.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora/sdxl/backpack.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/backpack.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full/sdxl/pink_sunglasses.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/pink_sunglasses.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora/sdxl/pink_sunglasses.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/pink_sunglasses.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full/sdxl/red_cartoon.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/full_ppl/sdxl/red_cartoon.yaml --override configs/subjects/full.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora/sdxl/red_cartoon.yaml --override configs/subjects/lora.yaml
	-@sleep 1
	-python -m src.scripts.train_lora --config configs/lora_ppl/sdxl/red_cartoon.yaml --override configs/subjects/lora.yaml
	-@sleep 1

train_test:
	-python -m src.scripts.train_lora --config configs/lora/sdxl/dog.yaml