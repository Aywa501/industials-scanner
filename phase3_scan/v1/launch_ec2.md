# Phase 3 CONUS scan — EC2 launch (v1)

Run the local steps first, then launch the box and follow the on-EC2 steps.

Prereqs (already done):
- S3 bucket `industrials-scanner-us-west-2` exists in us-west-2
- EC2 keypair created and `AWS_EC2_KEY_PATH` in `sites_us/.env` points to the `.pem`
- IAM user has `s3:Get*/Put*/List*` on your bucket and `s3:Get*/List*` on `sentinel-cogs`
- Service quota for `g6e` instance family is at least `4 vCPU` (g6e.xlarge = 4 vCPU)

## 1. Local: push bundle to S3

```
chmod +x sites_us/phase3_scan/v1/push_bundle.sh
./sites_us/phase3_scan/v1/push_bundle.sh
```

Uploads to `s3://industrials-scanner-us-west-2/scan-bundle/`:
- `phase3_grid.parquet` (~60 MB)
- `phase3_scenes.parquet` (~1 MB)
- `stage1_industrial_v1.pt` (~10 KB — head only; the ViT is downloaded from HF on the box)
- `bootstrap.sh`, `v1/infer_shard.py`, `.env`

## 2. Launch g6e.xlarge in us-west-2

Pick the latest **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)** in us-west-2.
A working AMI ID at the time of writing: `ami-04ec9a37e8e1dd3b1` — but **verify the latest** in the EC2 console under "Public images" filtered by `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)`.

```
aws ec2 run-instances \
  --region us-west-2 \
  --instance-type g6e.xlarge \
  --image-id <DLAMI_AMI_ID> \
  --key-name "$AWS_EC2_KEY_NAME" \
  --security-group-ids <YOUR_SG> \
  --iam-instance-profile Name=<YOUR_INSTANCE_PROFILE> \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=phase3-scan}]'
```

Use an **instance profile** (IAM role attached to the EC2) rather than putting AWS keys in `.env` — cleaner and avoids credential rotation pain. The role needs:
- `s3:GetObject` on `arn:aws:s3:::sentinel-cogs/*` (and `s3:ListBucket` on the bucket)
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on your `industrials-scanner-us-west-2` bucket

## 3. SSH and run

```
ssh -i "$AWS_EC2_KEY_PATH" ubuntu@<INSTANCE_PUBLIC_DNS>
```

Then on the box:

```
curl -O https://industrials-scanner-us-west-2.s3.us-west-2.amazonaws.com/scan-bundle/bootstrap.sh
BUCKET=industrials-scanner-us-west-2 nohup bash bootstrap.sh > scan.out 2>&1 &
tail -f scan.out
```

(`nohup` lets you disconnect; the scan continues. Tail to monitor.)

## 4. Monitor

The bootstrap loops over `~970` MGRS shards. Each shard prints a tiles/sec rate. On L40S in-region expect ~30–50 tiles/sec (vs 5/sec on local MPS), so the full scan should finish in **~15–25 hours** at a cost of **~$25–35** ($1.86/hr × wall time).

Per-shard results land in `data_us/phase3_results/{mgrs}.parquet` and are synced to S3 in batches. The scan is **resumable** — if the box dies, relaunch and `bootstrap.sh` skips already-done MGRS tiles.

## 5. Local: pull results and aggregate

When done:

```
aws s3 sync s3://industrials-scanner-us-west-2/phase3_results/ data_us/phase3_results/
python -m phase3_scan.v1.aggregate
```

Outputs `data_us/phase3_candidates.parquet` — DBSCAN-clustered candidate sites ranked by `max_prob × log(n_tiles)`.

## 6. Terminate the instance

Don't forget:

```
aws ec2 terminate-instances --instance-ids <INSTANCE_ID> --region us-west-2
```
