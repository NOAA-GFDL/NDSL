#!/usr/bin/env python3
import argparse
import json
import os

def addr_info(addr, bits, name):
    tmp = ''
    for i in range(bits//32):
        tmp += f'// 0x{addr+(i*4):02x} : Data signal of {name}\n'
        tmp += f'//        bit 31~0 - {name}[{((i+1)*32)-1}:{i*32}] (Read/Write)\n'
    tmp += f'// 0x{addr+bits//8:02x} : reserved\n'
    return tmp

def port(bits, name):
    return f'    output wire [{bits-1}:0] {name},\n'

def localparam_addr(addr_width, bits, addr, name):
    tmp = ''
    for i in range(bits//32):
        tmp += f'    ADDR_{name.upper()}_DATA_{i} = {addr_width}\'h{addr+(i*4):x},\n'
    tmp += f'    ADDR_{name.upper()}_CTRL   = {addr_width}\'h{addr+bits//8:x},\n'
    return tmp

def internal_reg(bits, name):
    return f'reg [{bits-1}:0] int_{name} = \'b0;\n'

def rdata(bits, name):
    tmp = ''
    for i in range(bits//32):
        tmp += f'                ADDR_{name.upper()}_DATA_{i}: begin\n'
        tmp += f'                    rdata <= int_{name}[{((i+1)*32)-1}:{i*32}];\n'
        tmp += f'                end\n'
    return tmp

def wdata(bits, name):
    tmp = ''
    for i in range(bits//32):
        tmp += f'// int_{name}[{((i+1)*32)-1}:{i*32}]\n'
        tmp += f'always @(posedge ACLK) begin\n'
        tmp += f'    if (ARESET)\n'
        tmp += f'        int_{name}[{((i+1)*32)-1}:{i*32}] <= 0;\n'
        tmp += f'    else if (ACLK_EN) begin\n'
        tmp += f'        if (w_hs && waddr == ADDR_{name.upper()}_DATA_{i})\n'
        tmp += f'            int_{name}[{((i+1)*32)-1}:{i*32}] <=\n'
        tmp += f'                (WDATA[31:0] & wmask) | (int_{name}[{((i+1)*32)-1}:{i*32}] & ~wmask);\n'
        tmp += f'    end\n'
        tmp += f'end\n'
        tmp += f'\n'
    return tmp

def reg_assign(name):
    return f'assign {name} = int_{name};\n'

# TODO C_S_AXI_ADDR_WIDTH
def control_module(name, ports, addr_infos, localparam_addrs, internal_regs, rdatas, wdatas, reg_assigns):
    return '''
`timescale 1ns/1ps
module {name}_control
#(parameter
    C_S_AXI_ADDR_WIDTH = 6,
    C_S_AXI_DATA_WIDTH = 32
)(
{ports}
    input  wire                            ACLK,
    input  wire                            ARESET,
    input  wire                            ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   AWADDR,
    input  wire                            AWVALID,
    output wire                            AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]   WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                            WVALID,
    output wire                            WREADY,
    output wire [1:0]                      BRESP,
    output wire                            BVALID,
    input  wire                            BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]   ARADDR,
    input  wire                            ARVALID,
    output wire                            ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]   RDATA,
    output wire [1:0]                      RRESP,
    output wire                            RVALID,
    input  wire                            RREADY,
    output wire                            interrupt,
    output wire                            ap_start,
    input  wire                            ap_done,
    input  wire                            ap_ready,
    input  wire                            ap_idle
);
//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
{addr_infos}
// (SC = Self Clear, COR = Clear on Read,
//  TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL = 6'h00,
    ADDR_GIE     = 6'h04,
    ADDR_IER     = 6'h08,
    ADDR_ISR     = 6'h0c,
{localparam_addrs}
    WRIDLE    = 2'd0,
    WRDATA    = 2'd1,
    WRRESP    = 2'd2,
    WRRESET   = 2'd3,
    RDIDLE    = 2'd0,
    RDDATA    = 2'd1,
    RDRESET   = 2'd2,
    ADDR_BITS = C_S_AXI_ADDR_WIDTH;

//------------------------Local signal-------------------
reg  [1:0]           wstate = WRRESET;
reg  [1:0]           wnext;
reg  [ADDR_BITS-1:0] waddr;
wire [31:0]          wmask;
wire                 aw_hs;
wire                 w_hs;
reg  [1:0]           rstate = RDRESET;
reg  [1:0]           rnext;
reg  [31:0]          rdata;
wire                 ar_hs;
wire [ADDR_BITS-1:0] raddr;

// internal registers
reg       int_ap_idle;
reg       int_ap_ready;
reg       int_ap_done = 1'b0;
reg       int_ap_start = 1'b0;
reg       int_auto_restart = 1'b0;
reg       int_gie = 1'b0;
reg [1:0] int_ier = 2'b0;
reg [1:0] int_isr = 2'b0;
{internal_regs}

//------------------------Instantiation------------------

//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = {{ {{8{{WSTRB[3]}}}}, {{8{{WSTRB[2]}}}}, {{8{{WSTRB[1]}}}}, {{8{{WSTRB[0]}}}} }};
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 1'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
{rdatas}
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt = int_gie & (|int_isr);
assign ap_start  = int_ap_start;
{reg_assigns}
// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (ap_done)
            int_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_ready <= ap_ready;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

{wdatas}

//------------------------Memory logic-------------------

endmodule
'''.format(name=name,
        ports=ports,
        addr_infos=addr_infos,
        localparam_addrs=localparam_addrs,
        internal_regs=internal_regs,
        rdatas=rdatas,
        wdatas=wdatas,
        reg_assigns=reg_assigns)

def generate_from_config(config):
    ports = ''
    addr_infos = ''
    localparam_addrs = ''
    internal_regs = ''
    rdatas = ''
    wdatas = ''
    reg_assigns = ''
    addr = 0x10
    params = [(name, bits) for name, bits in config['params']['scalars'].items()] + \
            [(name, 64) for name, _ in config['params']['memory'].items()]
    for name, bits in params:
        ports += port(bits, name)
        addr_infos += addr_info(addr, bits, name)
        localparam_addrs += localparam_addr(6, bits, addr, name)
        internal_regs += internal_reg(bits, name)
        rdatas += rdata(bits, name)
        wdatas += wdata(bits, name)
        reg_assigns += reg_assign(name)
        addr += bits//8 + 4

    return control_module(config['name'], ports, addr_infos, localparam_addrs, internal_regs, rdatas, wdatas, reg_assigns)

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

