#!/usr/bin/env bash
# Launch a tiny CPU spot in us-west-2 to pull Overture polygons (zero egress).
#   - c6a.xlarge (4 vCPU AMD, 8 GB) PRIMARY  — DuckDB wants ~few GB headroom
#   - m6a.large  (2 vCPU AMD, 8 GB) FALLBACK
#   - Ubuntu 22.04 (latest via SSM parameter)
set -uo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-096bbbe80d8ea7409 subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8)
TYPES=(r6a.2xlarge r6i.2xlarge r5.2xlarge r6a.4xlarge m6a.4xlarge)

# Push the bootstrap to the bundle prefix so the user-data can pull it.
echo "[launch-poly] pushing bootstrap_polygons.sh to s3://${BUCKET}/v3-pt2-bundle/"
aws s3 cp sites_us/phase2_classifier/v3_pt2/bootstrap_polygons.sh \
  "s3://${BUCKET}/v3-pt2-bundle/bootstrap_polygons.sh" \
  --region "$REGION" --only-show-errors

# Also push the latest fetch script via the existing bundle layout.
aws s3 cp sites_us/phase2_classifier/v3_pt2/fetch_candidate_polygons.py \
  "s3://${BUCKET}/v3-pt2-bundle/code/sites_us/phase2_classifier/v3_pt2/fetch_candidate_polygons.py" \
  --region "$REGION" --only-show-errors

AMI=$(aws ssm get-parameters --region "$REGION" \
  --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query 'Parameters[0].Value' --output text 2>/dev/null)
if [[ "$AMI" != ami-* ]]; then
  echo "[launch-poly] ERROR: could not resolve Ubuntu 22.04 AMI via SSM"; exit 1
fi
echo "[launch-poly] using AMI: $AMI"

UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
sudo apt-get update -qq && sudo apt-get install -y -qq awscli python3 python3-pip
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/v3-pt2-bundle/bootstrap_polygons.sh ./bootstrap_polygons.sh
  chmod +x bootstrap_polygons.sh
  BUCKET=${BUCKET} setsid nohup bash bootstrap_polygons.sh > /home/ubuntu/poly.out 2>&1 < /dev/null &
'
EOF

try_launch() {
  aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$1" \
    --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$2" \
    --iam-instance-profile "Name=$PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=stage2b-poly-fetch}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch-poly] attempt $attempt: $itype / $subnet (spot)"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch-poly] SUCCESS: $itype $out"
        echo "$out" > /tmp/poly_instance_id.txt
        echo
        echo "Stream log:  aws s3 cp s3://${BUCKET}/v3-pt2-artifacts/logs/${out}.poly.log - | tail -f"
        echo "Output:      s3://${BUCKET}/v3-pt2-artifacts/stage2_candidate_polygons.parquet"
        rm -f "$UD"
        exit 0
      fi
      echo "[launch-poly]   declined: $(echo "$out" | tr -d '\n' | cut -c1-200)"
    done
  done
  echo "[launch-poly] no spot capacity any type/AZ — waiting 300s (NEVER on-demand)"
  sleep 300
done
