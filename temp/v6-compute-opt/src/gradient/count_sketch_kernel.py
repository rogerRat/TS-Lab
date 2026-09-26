import triton
import triton.language as tl


@triton.jit
def count_sketch_kernel(output, values, hashes, signs, count, columns,
                        row_stride, column_stride, dimension: tl.constexpr,
                        BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    row = offsets // columns
    column = offsets % columns
    value = tl.load(values + row * row_stride + column * column_stride, valid, 0)
    sign = tl.load(signs + column, valid, 0)
    bucket = tl.load(hashes + column, valid, 0)
    tl.atomic_add(output + row * dimension + bucket, value * sign, valid, sem="relaxed")
