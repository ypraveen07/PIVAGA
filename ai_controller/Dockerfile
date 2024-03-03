FROM ultralytics/yolov5:v6.2
ADD ai_controller_v62 /home/
WORKDIR "/home/ai_controller_v62"
RUN pip install redis==4.3.4
RUN pip install bson==0.5.10
RUN pip install pymongo==4.2.0
CMD /opt/conda/bin/python worker_ups.py