# Deployment Failure Modes & Troubleshooting Guide

**Document ID**: `SPEC-DEPLOY-FAILURES`  
**Status**: `Approved`  
**Referenced By**: [05-production-deployment-cicd-spec.md](file:///home/ngthuan/projects/sales-assistant/specs/05-production-deployment-cicd-spec.md)  
**Target Architecture**: AWS EC2 (`t3.micro`/`t2.micro` Free Tier) + Amazon ECS (Bridge Networking) + ECR + GitHub Actions (OIDC)

---

## 1. Overview & Service Diagnostic Matrix

During CI/CD pipelines, container orchestration, and host provisioning on AWS Free Tier EC2 instances, errors can occur across different infrastructure layers. This document details each known failure mode, clearly identifying the **exact service or component having the error**, its root cause, diagnostic signals, and mitigation / recovery procedures.

| # | Error Signature / Message | Service Having Error | Subsystem / Layer | Root Cause Summary |
| :- | :--- | :--- | :--- | :--- |
| 1 | `already using a port required by your task` / `bind: address already in use` | **AWS ECS (Task Placement Engine)** & **Docker Host Engine** | Container Orchestration / Network Port Binding | Default rolling update launches task #2 before stopping task #1, colliding on host port `8000`. |
| 2 | `CannotPullContainerError: failed to extract layer ... no space left on device` | **Amazon EBS (Root Volume)** & **Docker Daemon** | Storage / Filesystem (`/dev/root`) | Default 8 GiB EBS volume runs out of free space during heavy image layer decompression. |
| 3 | `Exit Code 137` / Container Killed | **Linux OS Kernel (OOM Killer)** & **Docker Engine** | Host Memory Subsystem (`/proc/meminfo`) | Application and OS memory demands exceed physical 1 GiB RAM on `t3.micro` without swap. |
| 4 | EC2 Instance Not Registering in ECS Cluster | **Amazon ECS Container Agent (`amazon-ecs-agent`)** & **AWS IAM** | Host Agent / Cluster Registration | Missing cluster name in `/etc/ecs/ecs.config`, agent not running, or missing `ecsInstanceRole`. |
| 5 | Connection Timed Out on `http://<IP>:8000` | **AWS VPC (Security Groups)** | Ingress Network Firewall | Missing inbound rule allowing port `8000` on `sales-assistant-ec2-sg`, or no public IP. |
| 6 | `CannotPullContainerError: ...: not found` | **Amazon ECR (Elastic Container Registry)** | Container Image Registry | Image repository or tag `latest` does not exist in the target ECR registry. |
| 7 | `Container instance lacks required attribute com.amazonaws.ecs.capability.task-iam-role` | **Amazon ECS Container Agent (Credentials Proxy)** | Task Execution Agent Attributes | Self-managed EC2 instance lacks metadata credential proxy daemon required for task-level IAM roles. |
| 8 | `Could not assume role with OIDC: Not authorized to perform sts:AssumeRoleWithWebIdentity` | **AWS STS (Security Token Service)** & **AWS IAM OIDC Provider** | Authentication / Federation | Missing GitHub Actions root/intermediate CA thumbprints or `sub` claim mismatch in IAM trust policy. |
| 9 | `ResourceInitializationError: unable to retrieve secret` | **AWS Secrets Manager** & **AWS IAM (`ecsTaskExecutionRole`)** | Configuration & Secrets Management | Missing `secretsmanager:GetSecretValue` permission on `ecsTaskExecutionRole` or invalid secret name. |

---

## 2. Detailed Failure Cases & Runbooks

### Case 1: Host Port Conflict
- **Exact Service Having Error**: **AWS ECS (Task Placement Engine)** & **Docker Host Engine (EC2)**
- **Error Signature**:
  ```text
  RESOURCE:PORTS: [sales-assistant-task:8000] is already using a port required by your task
  docker: Error response from daemon: driver failed programming external connectivity on endpoint ... bind: address already in use.
  ```
- **Root Cause**:
  By default, ECS rolling deployments configure `minimumHealthyPercent = 100` and `maximumPercent = 200`. ECS attempts to start and healthcheck task revision *N+1* before terminating revision *N*. Because the service uses bridge networking with host port binding `8000:8000` on a single EC2 host, Docker cannot bind two containers to host port `8000` simultaneously.
- **Mitigation & Recovery Procedure**:
  1. In the ECS Service settings, expand **Deployment options** and set:
     - **Minimum running tasks in percent**: `0%`
     - **Maximum running tasks in percent**: `100%`
  2. Confirm service desired count is set strictly to `1`.
  3. Connect via SSM Session Manager and terminate orphaned containers or release port 8000:
     ```bash
     sudo docker ps
     sudo docker stop $(sudo docker ps -q --filter "ancestor=*sales-assistant*")
     ```
  4. If ECS agent task state is out of sync, restart the ECS agent:
     ```bash
     sudo systemctl restart ecs
     ```

---

### Case 2: Disk Exhaustion During Layer Extraction
- **Exact Service Having Error**: **Amazon EC2 / Amazon EBS (Elastic Block Store)** & **Docker Daemon**
- **Error Signature**:
  ```text
  CannotPullContainerError: failed to extract layer sha256:...: write /.../libtorch_cpu.so: no space left on device
  ```
- **Root Cause**:
  The default AWS EC2 launch template or quick-launch creates an 8 GiB root EBS volume. With the OS (~2.7 GiB) and a 2 GiB swapfile, only ~2.1 GiB of usable disk space remains. Large image layers (PyTorch, web scraping binaries, CUDA stubs) exceed the remaining space when decompressed under `/var/lib/docker`.
- **Mitigation & Recovery Procedure**:
  1. Ensure the EC2 root volume is provisioned as **30 GiB gp3** (fully covered under AWS Free Tier).
  2. On an existing EC2 instance:
     - In AWS EC2 Console > Elastic Block Store > Volumes: Select root volume > Actions > **Modify volume** > Set size to `30` GiB.
     - In instance shell via SSM, expand the partition and filesystem:
       ```bash
       sudo growpart /dev/nvme0n1 1
       # If ext4:
       sudo resize2fs /dev/root
       # If XFS:
       sudo xfs_growfs /
       ```
  3. Clean cached layers and broken partial downloads:
     ```bash
     sudo docker system prune -af --volumes
     ```

---

### Case 3: Out-Of-Memory (OOM) Termination (Exit Code 137)
- **Exact Service Having Error**: **Linux OS Kernel (OOM Killer)** & **Docker Engine on EC2**
- **Error Signature**:
  ```text
  Task stopped with exit code: 137 (SIGKILL by Out-Of-Memory killer)
  dmesg: "Out of memory: Killed process <pid> (python)"
  ```
- **Root Cause**:
  The `t3.micro` instance has only 1 GiB of physical RAM. Peak memory spikes (e.g. model initialization, concurrent ingestion, PyTorch CPU memory allocations) exceed 1 GiB. Without swap space, the Linux kernel invokes the OOM killer and immediately terminates the Python process.
- **Mitigation & Recovery Procedure**:
  1. Ensure a **2 GiB swapfile** is created and activated in EC2 User Data:
     ```bash
     sudo fallocate -l 2G /swapfile
     sudo chmod 600 /swapfile
     sudo mkswap /swapfile
     sudo swapon /swapfile
     echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
     ```
  2. Verify swap status: `free -m`.
  3. In the ECS Task Definition, configure soft reservation: `memoryReservation: 256` and omit rigid `memory` hard limits so Docker and the OS kernel can page idle memory to swap gracefully.

---

### Case 4: EC2 Instance Not Registering in ECS Cluster
- **Exact Service Having Error**: **Amazon ECS Container Agent (`amazon-ecs-agent`)** & **AWS IAM (`ecsInstanceRole`)**
- **Error Signature**:
  - In ECS Console, Cluster Container Instances tab shows **0 registered instances**.
  - `docker logs ecs-agent` reports:
    ```text
    [ERROR] Agent could not register with ECS: ClientException: Cluster not found
    # OR
    [ERROR] Unable to register as a container instance with ECS: NoCredentialProviders
    ```
- **Root Cause**:
  1. `/etc/ecs/ecs.config` is missing or contains the wrong cluster name.
  2. The EC2 instance was launched without the `ecsInstanceRole` IAM instance profile attached.
  3. The `amazon-ecs-init` / Docker daemon is not active.
- **Mitigation & Recovery Procedure**:
  1. Connect to EC2 via AWS Systems Manager (SSM) Session Manager.
  2. Verify and set `/etc/ecs/ecs.config`:
     ```bash
     echo "ECS_CLUSTER=sales-assistant-cluster" | sudo tee /etc/ecs/ecs.config
     ```
  3. Verify Docker and ECS services:
     ```bash
     sudo systemctl status docker
     sudo systemctl restart ecs
     sudo docker logs ecs-agent
     ```
  4. In AWS EC2 Console, ensure instance has IAM role `ecsInstanceRole` attached (with `AmazonEC2ContainerServiceforEC2Role` policy).

---

### Case 5: Connection Timed Out on Service Port
- **Exact Service Having Error**: **AWS VPC Security Groups (`sales-assistant-ec2-sg`)** & **EC2 Networking**
- **Error Signature**:
  ```text
  curl: (28) Failed to connect to <EC2_PUBLIC_IP> port 8000: Connection timed out
  ```
- **Root Cause**:
  1. The Security Group attached to the EC2 instance lacks an inbound rule permitting TCP traffic on port `8000`.
  2. The EC2 instance was launched in a public subnet without an Auto-assigned Public IPv4 address or an Elastic IP.
- **Mitigation & Recovery Procedure**:
  1. Open **VPC / EC2 Console > Security Groups > `sales-assistant-ec2-sg`**.
  2. Add Inbound Rule:
     - **Type**: Custom TCP
     - **Port Range**: `8000`
     - **Source**: Caller CIDR (or `0.0.0.0/0` for public testing).
  3. Confirm that the EC2 instance displays a valid **Public IPv4 address**.

---

### Case 6: Container Image Missing from ECR
- **Exact Service Having Error**: **Amazon ECR (Elastic Container Registry)**
- **Error Signature**:
  ```text
  CannotPullContainerError: inspect image has been retried 5 time(s): repository <ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/sales-assistant not found or image tag latest does not exist
  ```
- **Root Cause**:
  The ECS task definition references `<ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/sales-assistant:latest`, but the image repository has not been initialized or the image has not yet been pushed.
- **Mitigation & Recovery Procedure**:
  1. Verify the ECR repository exists:
     ```bash
     aws ecr describe-repositories --repository-names sales-assistant --region ap-southeast-1
     ```
  2. Trigger GitHub Actions deployment pipeline (`workflow_dispatch`), or push an initial container image manually:
     ```bash
     aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com
     docker build -t sales-assistant:latest .
     docker tag sales-assistant:latest <ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/sales-assistant:latest
     docker push <ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/sales-assistant:latest
     ```

---

### Case 7: Lacking Task IAM Role Capability
- **Exact Service Having Error**: **Amazon ECS Container Agent (Credentials Proxy)** & **Amazon Linux 2 / EC2 Host**
- **Error Signature**:
  ```text
  Container instance lacks required attribute com.amazonaws.ecs.capability.task-iam-role
  ```
- **Root Cause**:
  The Task Definition specified `taskRoleArn` on a self-managed EC2 instance where the ECS agent's credential proxy (redirecting metadata requests `169.254.170.2` via iptables) was not active or supported by the custom AMI configuration.
- **Mitigation & Recovery Procedure**:
  1. Leave **Task role** blank/empty in the ECS Task Definition so the container inherits permissions directly from the host's `ecsInstanceRole` via the standard EC2 instance metadata service (`http://169.254.169.254/latest/meta-data/`).
  2. Alternatively, enable the metadata proxy in `/etc/ecs/ecs.config`:
     ```bash
     echo "ECS_ENABLE_TASK_IAM_ROLE=true" | sudo tee -a /etc/ecs/ecs.config
     sudo systemctl restart ecs
     ```

---

### Case 8: OIDC Authentication & Role Assumption Failure
- **Exact Service Having Error**: **AWS STS (Security Token Service)** & **AWS IAM OIDC Identity Provider**
- **Error Signature**:
  ```text
  Error: Could not assume role with OIDC: Not authorized to perform sts:AssumeRoleWithWebIdentity
  ```
- **Root Cause**:
  1. The IAM OIDC Identity Provider (`token.actions.githubusercontent.com`) lacks the official GitHub root/intermediate CA thumbprints.
  2. The IAM Role trust policy condition for `token.actions.githubusercontent.com:sub` has casing or repository pattern discrepancies (e.g. `repo:logacc99/sales-assistant:ref:refs/heads/main` vs `repo:logacc99/sales-assistant:*`).
  3. The `Audience` in IAM OIDC provider does not match `sts.amazonaws.com`.
- **Mitigation & Recovery Procedure**:
  1. Update IAM OIDC provider thumbprints:
     ```bash
     aws iam update-open-id-connect-provider-thumbprint \
       --open-id-connect-provider-arn arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com \
       --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c5860f56b130439f181f4f0c69fa1aa14e0d6ff
     ```
  2. Ensure the Role Trust Policy matches:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Principal": {
             "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
           },
           "Action": "sts:AssumeRoleWithWebIdentity",
           "Condition": {
             "StringEquals": {
               "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
             },
             "StringLike": {
               "token.actions.githubusercontent.com:sub": "repo:logacc99/sales-assistant:*"
             }
           }
         }
       ]
     }
     ```

---

### Case 9: Task Execution Secret Retrieval Failure
- **Exact Service Having Error**: **AWS Secrets Manager** & **AWS IAM (`ecsTaskExecutionRole`)**
- **Error Signature**:
  ```text
  ResourceInitializationError: unable to retrieve secret: ...: AccessDeniedException: User: arn:aws:sts::<ACCOUNT_ID>:assumed-role/ecsTaskExecutionRole/... is not authorized to perform: secretsmanager:GetSecretValue on resource: arn:aws:secretsmanager:ap-southeast-1:<ACCOUNT_ID>:secret:sales-assistant/...
  ```
- **Root Cause**:
  When configuring container `secrets` in the ECS Task Definition, the `ecsTaskExecutionRole` (used by the ECS agent before launching the container) lacks permission to call `secretsmanager:GetSecretValue` on the target secret ARN.
- **Mitigation & Recovery Procedure**:
  1. Verify the secret exists in AWS Secrets Manager:
     ```bash
     aws secretsmanager describe-secret --secret-id sales-assistant/bedrock-api-key --region ap-southeast-1
     ```
  2. Attach an inline policy to `ecsTaskExecutionRole`:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "secretsmanager:GetSecretValue"
           ],
           "Resource": "arn:aws:secretsmanager:ap-southeast-1:<ACCOUNT_ID>:secret:sales-assistant/*"
         }
       ]
     }
     ```
