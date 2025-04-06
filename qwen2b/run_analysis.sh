for i in {0..20}; do
    python comp_test1.py --image $i --target_token "lesion"
    python comp_test1.py --image $i --target_token "lesion" --is_trained True --model_id darthPanda/qwen_2b_vl_radiology
done