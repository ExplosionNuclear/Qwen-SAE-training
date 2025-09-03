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

# Создаем очередь заданий
TASK_QUEUE=()
for model in "${MODELS[@]}"; do
    for hook_point in "${HOOK_POINTS[@]}"; do
        for layers in "${LAYER_GROUPS[@]}"; do
            TASK_QUEUE+=("$model|$hook_point|$layers")
        done
    done
done

echo "Всего заданий в очереди: ${#TASK_QUEUE[@]}"

# Количество GPU
NUM_GPUS=3
declare -a GPU_PIDS=(0 0 0)  # PID процессов на каждой GPU (0 = свободна)
TASK_INDEX=0

# Функция запуска задания на GPU
run_task_on_gpu() {
    local gpu_id=$1
    local task=$2
    
    IFS='|' read -r model hook_point layers <<< "$task"
    
    echo "GPU $gpu_id: Запускаем $model | $hook_point | $layers"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train_sae.py \
        --model_name="$model" \
        --hook_point="$hook_point" \
        --hook_point_layer="$layers" &
    
    local pid=$!
    GPU_PIDS[$gpu_id]=$pid
    echo "GPU $gpu_id: Запущен процесс PID=$pid"
}

# Основной цикл управления очередью
while [ $TASK_INDEX -lt ${#TASK_QUEUE[@]} ] || [ $(ps -p ${GPU_PIDS[@]} | grep -c python) -gt 0 ]; do
    
    # Проверяем каждую GPU
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        pid=${GPU_PIDS[$gpu_id]}
        
        # Если GPU свободна (процесс завершился или еще не запускался)
        if [ $pid -eq 0 ] || ! kill -0 $pid 2>/dev/null; then
            
            if [ $pid -ne 0 ]; then
                echo "GPU $gpu_id: Процесс PID=$pid завершен"
            fi
            
            # Если есть задания в очереди, запускаем следующее
            if [ $TASK_INDEX -lt ${#TASK_QUEUE[@]} ]; then
                task=${TASK_QUEUE[$TASK_INDEX]}
                run_task_on_gpu $gpu_id "$task"
                TASK_INDEX=$((TASK_INDEX + 1))
                echo "Осталось заданий: $((${#TASK_QUEUE[@]} - TASK_INDEX))"
            else
                GPU_PIDS[$gpu_id]=0
            fi
        fi
    done
    
    # Пауза перед следующей проверкой
    sleep 10
done

echo "Все задания завершены!"
