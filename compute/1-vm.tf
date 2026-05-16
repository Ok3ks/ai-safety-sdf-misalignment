variable "user_data" {
  default = <<EOT
users:
 - name: ubuntu
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHM+JKOzaxLPBmqzGyTxfQ4STMS+yQDwVNJYLVxwmR0u freelanceokeks@gmail.com
  EOT
}

resource "nebius_compute_v1_disk" "boot_disk" {
  name = var.boot_disk_name
  parent_id = "project-e00t3jrnpr0043nvspd4qw"
  type = "NETWORK_SSD"
  block_size_bytes = 4096
  size_bytes = 579820584960
  source_image_family = {
    image_family = var.boot_disk_image_family
  }
}

resource "nebius_compute_v1_instance" "vm" {
  name = "green-clownfish-instance-3"
  parent_id = "project-e00t3jrnpr0043nvspd4qw"
  stopped = false
  resources = {
    platform = var.vm_platform
    preset = var.vm_preset
  }
  boot_disk = {
    existing_disk = {
      id = nebius_compute_v1_disk.boot_disk.id
    }
    attach_mode = "READ_WRITE"
    device_id = "boot-disk"
  }
  network_interfaces = [
    {
      name = "eth0"
      subnet_id = "vpcsubnet-e00enzr2n2nja43wq7"
      ip_address = {}
      public_ip_address = {
        static = true
      }
    }
  ]
  
  cloud_init_user_data = var.user_data
  reservation_policy = {
    policy = "AUTO"
  }
}

output "vm_ip" {
  value       = nebius_compute_v1_instance.vm.network_interfaces[0].public_ip_address
  description = "IP of created VM"
}