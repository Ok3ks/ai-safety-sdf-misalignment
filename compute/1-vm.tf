variable "user_data" {
  default = <<EOT
users:
 - name: ${var.ssh_user}
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ${trimspace(file(var.ssh_public_key_path))}
  EOT
}

resource "nebius_compute_v1_disk" "boot_disk" {
  name = var.boot_disk_name
  parent_id = var.parent_id
  type = "NETWORK_SSD"
  block_size_bytes = 4096
  size_bytes = 107374182400
  source_image_family = {
    image_family = "ubuntu24.04-driverless"
  }
  disk_encryption = {
    type = "DISK_ENCRYPTION_UNSPECIFIED"
  }
}

resource "nebius_compute_v1_instance" "vm" {
  name = var.vm_name
  parent_id = var.parent_id
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
      subnet_id = var.subnet_id
      ip_address = {"allocationId":""}
      public_ip_address = {
        static = true
      }
    }
  ]
  service_account = var.service_account_id
  cloud_init_user_data = var.user_data
  reservation_policy = {
    policy = "AUTO"
  }
}

