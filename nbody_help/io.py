import re
import numpy as np 

## Some structures
gadget2_header_dtype = np.dtype([
    ('np', np.int32, 6),
    ("mass", np.float64, 6),
    ("time", np.float64),
    ("redshift", np.float64),
    ("flag_sfr", np.int32),
    ("flag_feedback", np.int32),
    ("np_total", np.uint32, 6),
    ("flag_cooling", np.int32),
    ("num_files", np.int32),
    ("boxsize", np.float64),
    ("omega0", np.float64),
    ("omegaLambda", np.float64),
    ("hubble_param", np.float64),
    ("flag_stellarage", np.int32),
    ("flag_metals", np.int32),
    ("np_total_highword", np.uint32, 6),
    ("flag_entropy_instead_u", np.int32),
    ("fill", np.bytes_, 60)
    ], align=True)

## Some assistant functions
def analyze_float_format(s):
    """
    更精确地分析浮点数字符串的格式
    """
    value = float(s)
    
    # 科学计数法模式
    sci_pattern_E = r'^([-+]?)(\d+)(?:\.(\d+))?[E]([-+]?\d+)$'
    sci_pattern_e = r'^([-+]?)(\d+)(?:\.(\d+))?[e]([-+]?\d+)$'
    # 普通小数模式
    decimal_pattern = r'^([-+]?)(\d*)(?:\.(\d+))?$'
    
    sci_match_e = re.match(sci_pattern_e, s)
    sci_match_E = re.match(sci_pattern_E, s)
    decimal_match = re.match(decimal_pattern, s)
    
    if sci_match_e or sci_match_E:
        # 科学计数法
        if sci_match_e:
            sign, int_part, frac_part, exp = sci_match_e.groups()
            format_str = "e"
        elif sci_match_E:
            sign, int_part, frac_part, exp = sci_match_E.groups()
            format_str = "E"
        if frac_part:
            precision = len(frac_part)
            format_spec = f".{precision}{format_str}"
        else:
            format_spec = f".0{format_str}"
    elif decimal_match:
        # 普通小数
        sign, int_part, frac_part = decimal_match.groups()
        if frac_part:
            precision = len(frac_part)
            format_spec = f".{precision}f"
        else:
            format_spec = ".0f"
    else:
        format_spec = ".6f"  # 默认格式
    
    return value, format_spec
    

def _read_single_gadget2_file(args):
    """Read a single gadget2 file. This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (filename, use_long_int, only_pos, sub_rate, file_index)
    
    Returns:
        Tuple of (file_index, header, pos, vel, id)
    """
    filename, use_long_int, only_pos, sub_rate, file_index = args
    
    f = open(filename, "rb")
    try:
        header_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        header = np.fromfile(f, dtype=gadget2_header_dtype, count=1)[0]
        header_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if header_flag_1 != header_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(header_flag not consistent: {header_flag_1} != {header_flag_2})")
        npar = header["np"][1]
        pos_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        pos_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
        pos_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        if pos_flag_1 != pos_flag_2:
            raise ValueError(f"file {filename} is not a gadget2 snapshot(pos_flag not consistent: {pos_flag_1} != {pos_flag_2})")
        if not only_pos:
            vel_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            vel_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
            vel_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
            if vel_flag_1 != vel_flag_2:
                raise ValueError(f"file {filename} is not a gadget2 snapshot(vel_flag not consistent: {vel_flag_1} != {vel_flag_2})")
            id_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            id_temp = np.fromfile(f, dtype=np.uint64 if use_long_int else np.uint32, count=npar)
            id_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
            if id_flag_1 != id_flag_2:
                raise ValueError(f"file {filename} is not a gadget2 snapshot(id_flag not consistent: {id_flag_1} != {id_flag_2})")
        else:
            vel_temp = None
            id_temp = None

        if sub_rate < 1.0:
            index_choose = np.random.choice(npar, int(npar*sub_rate), replace=False)
            pos_temp = pos_temp[index_choose]
            if not only_pos:
                vel_temp = vel_temp[index_choose]
                id_temp = id_temp[index_choose]
        
        return (file_index, header, pos_temp, vel_temp, id_temp)
    finally:
        f.close()


def read_gadget2(filename=None, filenames_list=None, use_long_int=False, only_pos=False, sub_rate=1.0, show_progress=False, nthreads=1):
    """Read gadget2 snapshot files.
    
    Args:
        filename: Single filename (mutually exclusive with filenames_list)
        filenames_list: List of filenames to read
        use_long_int: If True, use int64 for particle IDs
        only_pos: If True, only read positions (skip velocity and ID)
        sub_rate: Subsampling rate (0.0 to 1.0). If < 1.0, randomly sample particles
        show_progress: If True, show progress bar (only for single-threaded mode)
        nthreads: Number of threads for parallel reading. Default is 1 (sequential)
    """
    if filename is None and filenames_list is None:
        raise ValueError("Either filename or filenames_list must be provided")
    if filename is not None:
        filenames_list = [filename, ]
    
    num_files = len(filenames_list)
    
    # Force single-threaded when only one file to avoid thread overhead
    if num_files <= 1:
        nthreads = 1
    
    # Single-threaded mode (original behavior)
    if nthreads <= 1:
        headers_array = np.empty(num_files, dtype=gadget2_header_dtype)
        pos_list = []
        vel_list = []
        id_list = []

        if show_progress:
            import tqdm 
            iterator = tqdm.tqdm(enumerate(filenames_list), total=num_files, desc="Reading gadget2")
        else:
            iterator = enumerate(filenames_list)

        for i, fname in iterator:
            f = open(fname, "rb")
            header_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            headers_array[i] = np.fromfile(f, dtype=gadget2_header_dtype, count=1)[0]
            header_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
            if header_flag_1 != header_flag_2:
                raise ValueError(f"file {fname} is not a gadget2 snapshot(header_flag not consistent: {header_flag_1} != {header_flag_2})")
            npar = headers_array[i]["np"][1]
            pos_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
            pos_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
            pos_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
            if pos_flag_1 != pos_flag_2:
                raise ValueError(f"file {fname} is not a gadget2 snapshot(pos_flag not consistent: {pos_flag_1} != {pos_flag_2})")
            if not only_pos:
                vel_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
                vel_temp = np.fromfile(f, dtype=np.float32, count=npar*3).reshape(npar, 3)
                vel_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
                if vel_flag_1 != vel_flag_2:
                    raise ValueError(f"file {fname} is not a gadget2 snapshot(vel_flag not consistent: {vel_flag_1} != {vel_flag_2})")
                id_flag_1 = np.fromfile(f, dtype=np.int32, count=1)[0]
                id_temp = np.fromfile(f, dtype=np.uint64 if use_long_int else np.uint32, count=npar)
                id_flag_2 = np.fromfile(f, dtype=np.int32, count=1)[0]
                if id_flag_1 != id_flag_2:
                    raise ValueError(f"file {fname} is not a gadget2 snapshot(id_flag not consistent: {id_flag_1} != {id_flag_2})")
            else:
                vel_temp = None 
                id_temp = None

            if sub_rate < 1.0:
                index_choose = np.random.choice(npar, int(npar*sub_rate), replace=False)
                pos_temp = pos_temp[index_choose]
                if not only_pos:
                    vel_temp = vel_temp[index_choose]
                    id_temp = id_temp[index_choose]
            
            pos_list.append(pos_temp)
            vel_list.append(vel_temp)
            id_list.append(id_temp)
        
        pos_array = np.concatenate(pos_list, axis=0)
        if not only_pos:
            vel_array = np.concatenate(vel_list, axis=0)
            id_array = np.concatenate(id_list, axis=0)
        else:
            vel_array = None 
            id_array = None
        if headers_array.shape[0] == 1:
            headers_array = headers_array[0]
        
        return {
            "pos": pos_array,
            "vel": vel_array,
            "id": id_array,
            "header": headers_array,
        }
    
    # Multi-threaded mode
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Prepare arguments for each file
    args_list = [(fn, use_long_int, only_pos, sub_rate, i) for i, fn in enumerate(filenames_list)]
    
    # Initialize result arrays
    headers_array = np.empty(num_files, dtype=gadget2_header_dtype)
    pos_list = [None] * num_files
    vel_list = [None] * num_files
    id_list = [None] * num_files
    
    # Use ThreadPoolExecutor for parallel reading
    # Note: GIL is released during I/O operations, so threads provide speedup for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(_read_single_gadget2_file, args): args[4] for args in args_list}
        
        # Collect results with optional progress bar
        if show_progress:
            import tqdm
            futures = tqdm.tqdm(as_completed(future_to_idx), total=num_files, desc="Reading gadget2")
        else:
            futures = as_completed(future_to_idx)
        
        for future in futures:
            file_idx, header, pos, vel, ids = future.result()
            headers_array[file_idx] = header
            pos_list[file_idx] = pos
            vel_list[file_idx] = vel
            id_list[file_idx] = ids
    
    # Concatenate results in order
    pos_array = np.concatenate(pos_list, axis=0)
    if not only_pos:
        vel_array = np.concatenate(vel_list, axis=0)
        id_array = np.concatenate(id_list, axis=0)
    else:
        vel_array = None
        id_array = None
    
    if headers_array.shape[0] == 1:
        headers_array = headers_array[0]
    
    return {
        "pos": pos_array,
        "vel": vel_array,
        "id": id_array,
        "header": headers_array,
    }

def read_rockstar(filename, need_elements = ["pos", "vel", "Rvir", "Mvir"], num_density=None, boxsize=1000.0):
    """
    Args:
        num_density (float): If set, will select a subset of particles with density \approx num_density. Default is None. The result ndarray must include "pos" and "Mvir".
    """
    supported_elements = ["pos", "vel", "Rvir", "Mvir"]
    if not set(need_elements).issubset(supported_elements):
        raise ValueError(f"Elements {need_elements} are not supported")
    source = np.loadtxt(filename, usecols=[8, 9, 10, 11, 12, 13, 2, 5], dtype=np.float32)

    if num_density is not None:
        if source.shape[0] < boxsize ** 3 * num_density:
            raise ValueError(f"num_density {num_density} is too large. Now only {source.shape[0] / boxsize ** 3:.3E}. The number of halo must be smaller than {boxsize ** 3 * num_density:.3E}.")
        else:
            if "pos" not in need_elements:
                need_elements.append("pos")
            if "Mvir" not in need_elements:
                need_elements.append("Mvir")

    dtype_list = []
    if "pos" in need_elements:
        dtype_list.append(("pos", np.float32, 3))
    if "vel" in need_elements:
        dtype_list.append(("vel", np.float32, 3))
    if "Rvir" in need_elements:
        dtype_list.append(("Rvir", np.float32))
    if "Mvir" in need_elements:
        dtype_list.append(("Mvir", np.float32))
    npy_dtype = np.dtype(dtype_list)

    result_npy = np.empty(shape=(len(source),), dtype=npy_dtype)
    for need_element in need_elements:
        if need_element == "pos":
            result_npy["pos"] = source[:, :3]
        if need_element == "vel":
            result_npy["vel"] = source[:, 3:6]
        if need_element == "Rvir":
            result_npy["Rvir"] = source[:, 6]
        if need_element == "Mvir":
            result_npy["Mvir"] = source[:, 7]

    if num_density is not None:
        result_npy.sort(order="Mvir")
        result_npy = result_npy[::-1]
        min_num = int(boxsize ** 3 * num_density)
        base_mass = result_npy["Mvir"][min_num - 1]
        iter_mass = result_npy["Mvir"][min_num]
        while iter_mass >= base_mass:
            min_num += 1
            if min_num >= len(result_npy):
                break
            iter_mass = result_npy["Mvir"][min_num]
        result_npy = result_npy[:min_num] 

    return result_npy
