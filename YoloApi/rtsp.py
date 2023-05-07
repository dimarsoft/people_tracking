from django.http import JsonResponse

from YoloApi import settings
from rtsp import RtspStreamReaderToFile, init_cv

# пока хардкор
rtsp_app = RtspStreamReaderToFile("rtsp://stream:ShAn675sb5@31.173.67.209:13554", "camera1", settings.VIDEO_ROOT)
init_cv()


def stop_rtsp(request):
    rtsp_app.stop()

    if request.method == 'POST':
        return JsonResponse(
            {
                'status': 'success',
                "file_id": "100"
            }
        )


def start_rtsp(request):
    rtsp_app.start()

    if request.method == 'POST':
        rtsp_url: str = "Привет"
        return JsonResponse(
            {
                'status': 'success',
                "rtsp_url": rtsp_url
            }
        )
