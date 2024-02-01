#!/usr/bin/env python3
import argparse
import json
import os

def create(name, vendor, version, module_name):
    return f'create_ip -name {name} -vendor {vendor} -library ip -version {version} -module_name {module_name}\n'

def set_params(params, module_name):
    tmp = 'set_property -dict [list'
    for key, value in params.items():
        tmp += f' {key} {{{value}}}'
    tmp += f'] [get_ips {module_name}]\n'
    return tmp

def synth_script(ip_cores):
    synth_ip = 'synth_ip [get_ips]' if ip_cores != '' else ''
    return '''
if {{ !($::argc == 7 || $::argc == 8) }} {{
    puts "Error: Program \\"$::argv0\\" requires 7-8 arguments.\\n"
    puts "Usage: $::argv0 <src_dir> <top_file> <build_dir> <lib_dir> <gen_dir> <user_ip_repo> <board_part> (elaborate)\\n"
    exit 1
}}

set src_dir    [lindex $::argv 0]
set top_file   [lindex $::argv 1]
set build_dir  [lindex $::argv 2]
set lib_dir    [lindex $::argv 3]
set gen_dir    [lindex $::argv 4]
set user_repo  [lindex $::argv 5]
set board_part [lindex $::argv 6]

create_project batch_synthesis $build_dir/synthesis -part $board_part -force
add_files [glob $src_dir/*.*v $lib_dir/*.*v $gen_dir/*.*v]
set_property top $top_file [current_fileset]
set_property top_file {{$src_dir/$top_file}} [current_fileset]
if {{$user_repo != ""}} {{
    set_property ip_repo_paths $user_repo [current_project]
    update_ip_catalog -rebuild
}}
{ip_cores}
update_compile_order -fileset sources_1
update_compile_order -fileset sources_1
set_msg_config -id "HDL" -new_severity "ERROR"
check_syntax
reset_msg_config -id "HDL" -default_severity
{synth_ip}
if {{ $::argc == 7 }} {{
    synth_design -top $top_file -rtl
}} else {{
    synth_design -top $top_file
}}

close_project
'''.format(ip_cores=ip_cores, synth_ip=synth_ip)

def generate_from_config(config):
    ip_cores = ''
    for module_name, info in config['ip_cores'].items():
        ip_cores += create(info['name'], info['vendor'], info['version'], module_name)
        ip_cores += set_params(info['params'], module_name)

    return synth_script(ip_cores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating package tcl script')

    parser.add_argument('config', nargs=1, help='The config file describing the core')
    parser.add_argument('-o', '--output', help='The output path for the resulting tcl script', metavar='<file>', nargs=1, default=['package_kernel.tcl'])
    parser.add_argument('-f', '--force', help='Toggles whether output file should be overwritten', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.config[0]):
        print (f'Error, {args.config} does not exist')
        quit(1)
    with open(args.config[0], 'r') as f:
        config = json.load(f)

    file_str = generate_from_config(config)

    if not args.force and os.path.exists(args.output[0]):
        print (f'Error, "{args.output[0]}" already exists. Add -f flag to overwrite')
        quit(1)
    with open(args.output[0], 'w') as f:
        f.write(file_str)

