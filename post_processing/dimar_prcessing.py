from post_processing.functions import crossing_bound, get_deviations, process_filt
from post_processing.timur import tracks_to_dic, get_camera
from tools.count_results import Result, Deviation
from tools.labeltools import get_status


def dimar_count_humans(tracks, source, bound_line, log: bool = True) -> Result:
    """
    Постобработка на основе кода Тимура, но с некоторыми изменениями.
    Parameters
    ----------
    tracks
    source
    bound_line
    log

    Returns
    -------

    """
    print(f"Dmitrii postprocessing v1.2_01.05.2023")

    camera_num, w, h, fps = get_camera(source)

    if log:
        print(f"camera_num =  {camera_num}, ({w} {h})")

    people_tracks, helmet_tracks, vest_tracks = tracks_to_dic(tracks, w, h)

    if len(people_tracks) == 0:
        return Result(0, 0, 0, [])

    people_tracks = process_filt(people_tracks)

    if len(people_tracks) == 0:
        return Result(0, 0, 0, [])

    if log:
        print(f"bound_line =  {bound_line}")

    tracks_info = []
    for p_id in people_tracks.keys():
        people_path = people_tracks[p_id]
        tr_info = crossing_bound(people_path['path'], bound_line)
        tracks_info.append(tr_info)
        if log:
            print(f"{p_id}: {tr_info}")

    deviations = []

    deviations_info, result = get_deviations(people_tracks, helmet_tracks, vest_tracks, bound_line,
                                             helmet_or_uniform_conf=0.1,
                                             log=log)

    count_in = result["input"]
    count_out = result["output"]

    # print(deviations_info)

    for item in deviations_info:
        frame_id = item["frame_id"]

        start_frame = item["start_frame"]
        end_frame = item["end_frame"]

        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        # -+ 1 сек от пересечения, но не забегая за границы человека по треку
        start_frame = max(frame_id - 2 * fps, start_frame)
        end_frame = min(frame_id + 2 * fps, end_frame)

        if start_frame <= frame_id <= end_frame:
            pass
        else:
            # для проверки
            print(f"bad: {start_frame}, {frame_id}, {end_frame}")

        status = get_status(item["has_helmet"], item["has_uniform"])

        if status > 0:  # 0 нет нарушения
            deviations.append(Deviation(start_frame, end_frame, status))

    if log:
        print(f"{camera_num}: count_in = {count_in}, count_out = {count_out}, deviations = {len(deviations)}")

    return Result(count_in + count_out, count_in, count_out, deviations)
