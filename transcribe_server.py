if __name__ == "__main__":
    from transcribe_demo import get_argparse_args, do_it, get_only_updated_segments, get_last_segments
    import logging
    import json
    import datetime
    import time

    from server_broadcasting_py import Server

    logger = logging.getLogger(__name__)

    HOST = "localhost"
    PORT = 7868
    socket_address = (HOST, PORT)

    DEBUG = True
    sleep_time = 0
    """
    Don't process continuously, sleep a bit
    Infinite loops are bad for processors
    """

    server = Server(socket_address)
    server.start()

    args = get_argparse_args()

    r = do_it(args)
    if not r:
        exit()

    (
        transcription_queue,
        completed_segments,
        audio_recorder_thread,
        transcription_thread,
    ) = r

    while True:
        try:
            if len(server.clients) > 0 or DEBUG:
                tr_update = transcription_queue.get()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                
                data_to_broadcast = {
                    **tr_update,
                    "timestamp": timestamp,
                }
                json_str = json.dumps(data_to_broadcast)
                print(f"json_str\n{len(json_str)}\n{json_str}")
                logger.info(json_str)
                server.broadcast(data_to_broadcast)

                transcription_queue.task_done()
            # Infinite loops are bad for processors
            time.sleep(sleep_time)
        except Exception as e:
            logger.exception(e)
