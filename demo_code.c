static void gen_exception(DisasContext *s, uint32_t where, int nr)
{
    gen_flush_cc_op(s);
    gen_jmp_im(s, where);
    gen_helper_raise_exception(tcg_const_i32(nr));
}

static inline void gen_addr_fault(DisasContext *s)
{
    gen_exception(s, s->insn_pc, EXCP_ADDRESS);
}

#define SRC_EA(result, opsize, op_sign, addrp) do { \
    result = gen_ea(s, insn, opsize, NULL_QREG, addrp, op_sign ? EA_LOADS : EA_LOADU); \
    if (IS_NULL_QREG(result)) { \
        gen_addr_fault(s); \
        return; \
    } \
    } while (0)

#define DEST_EA(insn, opsize, val, addrp) do { \
    TCGv ea_result = gen_ea(s, insn, opsize, val, addrp, EA_STORE); \
    if (IS_NULL_QREG(ea_result)) { \
        gen_addr_fault(s); \
        return; \
    } \
    } while (0)

/* Generate a jump to an immediate address.  */
static void gen_jmp_tb(DisasContext *s, int n, uint32_t dest)
{
    TranslationBlock *tb;

    tb = s->tb;
    if (unlikely(s->singlestep_enabled)) {
        gen_exception(s, dest, EXCP_DEBUG);
    } else if ((tb->pc & TARGET_PAGE_MASK) == (dest & TARGET_PAGE_MASK) ||
               (s->pc & TARGET_PAGE_MASK) == (dest & TARGET_PAGE_MASK)) {
        tcg_gen_goto_tb(n);
        tcg_gen_movi_i32(QREG_PC, dest);
        tcg_gen_exit_tb((long)tb + n);
    } else {
        gen_jmp_im(s, dest);
        tcg_gen_exit_tb(0);
    }
    s->is_jmp = DISAS_TB_JUMP;
}

DISAS_INSN(undef_mac)
{
    gen_exception(s, s->pc - 2, EXCP_LINEA);
}

DISAS_INSN(undef_fpu)
{
    gen_exception(s, s->pc - 2, EXCP_LINEF);
}

DISAS_INSN(undef)
{
    gen_exception(s, s->pc - 2, EXCP_UNSUPPORTED);
    cpu_abort(cpu_single_env, "Illegal instruction: %04x @ %08x",
              insn, s->pc - 2);
}

DISAS_INSN(mulw)
{
    TCGv reg;
    TCGv tmp;
    TCGv src;
    int sign;

    sign = (insn & 0x100) != 0;
    reg = DREG(insn, 9);
    tmp = tcg_temp_new();
    if (sign)
        tcg_gen_ext16s_i32(tmp, reg);
    else
        tcg_gen_ext16u_i32(tmp, reg);
    SRC_EA(src, OS_WORD, sign, NULL);
    tcg_gen_mul_i32(tmp, tmp, src);
    tcg_gen_mov_i32(reg, tmp);
    /* Unlike m68k, coldfire always clears the overflow bit.  */
    gen_logic_cc(s, tmp);
}

DISAS_INSN(divw)
{
    TCGv reg;
    TCGv tmp;
    TCGv src;
    int sign;

    sign = (insn & 0x100) != 0;
    reg = DREG(insn, 9);
    if (sign) {
        tcg_gen_ext16s_i32(QREG_DIV1, reg);
    } else {
        tcg_gen_ext16u_i32(QREG_DIV1, reg);
    }
    SRC_EA(src, OS_WORD, sign, NULL);
    tcg_gen_mov_i32(QREG_DIV2, src);
    if (sign) {
        gen_helper_divs(cpu_env, tcg_const_i32(1));
    } else {
        gen_helper_divu(cpu_env, tcg_const_i32(1));
    }

    tmp = tcg_temp_new();
    src = tcg_temp_new();
    tcg_gen_ext16u_i32(tmp, QREG_DIV1);
    tcg_gen_shli_i32(src, QREG_DIV2, 16);
    tcg_gen_or_i32(reg, tmp, src);
    s->cc_op = CC_OP_FLAGS;
}

DISAS_INSN(divl)
{
    TCGv num;
    TCGv den;
    TCGv reg;
    uint16_t ext;

    ext = lduw_code(s->pc);
    s->pc += 2;
    if (ext & 0x87f8) {
        gen_exception(s, s->pc - 4, EXCP_UNSUPPORTED);
        return;
    }
    num = DREG(ext, 12);
    reg = DREG(ext, 0);
    tcg_gen_mov_i32(QREG_DIV1, num);
    SRC_EA(den, OS_LONG, 0, NULL);
    tcg_gen_mov_i32(QREG_DIV2, den);
    if (ext & 0x0800) {
        gen_helper_divs(cpu_env, tcg_const_i32(0));
    } else {
        gen_helper_divu(cpu_env, tcg_const_i32(0));
    }
    if ((ext & 7) == ((ext >> 12) & 7)) {
        /* div */
        tcg_gen_mov_i32 (reg, QREG_DIV1);
    } else {
        /* rem */
        tcg_gen_mov_i32 (reg, QREG_DIV2);
    }
    s->cc_op = CC_OP_FLAGS;
}

DISAS_INSN(addsub)
{
    TCGv reg;
    TCGv dest;
    TCGv src;
    TCGv tmp;
    TCGv addr;
    int add;

    add = (insn & 0x4000) != 0;
    reg = DREG(insn, 9);
    dest = tcg_temp_new();
    if (insn & 0x100) {
        SRC_EA(tmp, OS_LONG, 0, &addr);
        src = reg;
    } else {
        tmp = reg;
        SRC_EA(src, OS_LONG, 0, NULL);
    }
    if (add) {
        tcg_gen_add_i32(dest, tmp, src);
        gen_helper_xflag_lt(QREG_CC_X, dest, src);
        s->cc_op = CC_OP_ADD;
    } else {
        gen_helper_xflag_lt(QREG_CC_X, tmp, src);
        tcg_gen_sub_i32(dest, tmp, src);
        s->cc_op = CC_OP_SUB;
    }
    gen_update_cc_add(dest, src);
    if (insn & 0x100) {
        DEST_EA(insn, OS_LONG, dest, &addr);
    } else {
        tcg_gen_mov_i32(reg, dest);
    }
}


/* Reverse the order of the bits in REG.  */
DISAS_INSN(bitrev)
{
    TCGv reg;
    reg = DREG(insn, 0);
    gen_helper_bitrev(reg, reg);
}