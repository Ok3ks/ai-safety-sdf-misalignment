variable "parent_id" {
    type = string
}

variable "subnet_id" {
    type = string
}

variable "service_account_id" {
    type = string
}

variable "vm_platform" {
    type = string
}

variable "vm_preset" {
    type = string
}

variable "ssh_user" {
    type = string
    default = "ubuntu"
}

variable "boot_disk_name" {
    type=string
    default="ubuntu"
}

variable "boot_disk_image_family" {
    type=string
}

variable "ssh_public_key_path" {
    type = string
    default = "~/.ssh/id_ed25519.pub"
}