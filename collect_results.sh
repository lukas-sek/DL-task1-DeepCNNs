BSC_USER="nct01204"
BSC_HOST="alogin1.bsc.es"
REMOTE_DIR="~/DL-Lab-CNN"

echo "Pulling results from BSC ..."

# results.json and config.json for every run
for exp in std_underfit std_overfit std_best res_underfit res_overfit res_best transfer_frozen transfer_finetune; do
    echo "  $exp ..."
    scp "${BSC_USER}@${BSC_HOST}:${REMOTE_DIR}/runs/${exp}/results.json" \
        "runs/${exp}/results.json" 2>/dev/null && echo "    OK" || echo "    not found (job not run yet)"
    scp "${BSC_USER}@${BSC_HOST}:${REMOTE_DIR}/runs/${exp}/config.json" \
        "runs/${exp}/config.json" 2>/dev/null
done

# SLURM logs
mkdir -p logs_bsc
scp "${BSC_USER}@${BSC_HOST}:${REMOTE_DIR}/logs/*.out" logs_bsc/ 2>/dev/null

echo ""
echo "Run python plot_results.py --runs runs/ --out_dir figures/ to generate plots."
