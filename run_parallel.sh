#!/bin/bash

# Модели
MODELS=(
    "ExplosionNuclear/Llama-2.3-3B-Instruct-special"
    "ExplosionNuclear/Llama-2.3-3B-Instruct-special-merged-with-19-exp"
)

# Типы хуков  
HOOK_POINTS=(
    "blocks.{layer}.hook_resid_pre"
    "blocks.{layer}.hook_mlp_in"
)

# Группы слоев (от 0 до 27 с шагом 6)
LAYER_GROUPS=(
    "[0,1,2,3,4,5]"
    "[6,7,8,9,10,11]" 
    "[12,13,14,15,16,17]"
    "[18,19,20,21,22,23]"
    "[24,25,26,27]"
)

gpu_counter=0

# Запускаем все комбинации
for model in "${MODELS[@]}"; do
    for hook_point in "${HOOK_POINTS[@]}"; do
        for layers in "${LAYER_GROUPS[@]}"; do
            gpu_id=$((gpu_counter % 3))  # Используем GPU 0, 1, 2 циклически
            
            echo "GPU $gpu_id: Запускаем $model | $hook_point | $layers"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python train_sae.py \
                --model_name="$model" \
                --hook_point="$hook_point" \
                --hook_point_layer="$layers" &
            
            gpu_counter=$((gpu_counter + 1))
            sleep 2  # Небольшая задержка между запусками
        done
    done
done

echo "Все процессы запущены. Ожидаем завершения..."
wait
echo "Все обучения завершены!"
