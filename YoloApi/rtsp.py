from django.http import JsonResponse


def stop_rtsp(request):
    if request.method == 'POST':
        return JsonResponse(
            {
                'status': 'success',
                "file_id": "100"
            }
            )

def start_rtsp(request):
    if request.method == 'POST':
        rtsp_url: str = "Привет"
        return JsonResponse(
            {
                'status': 'success',
                "rtsp_url": rtsp_url
            }
            )
    #else:
    #    # Return a 405 Method Not Allowed error for other HTTP methods
    #    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
