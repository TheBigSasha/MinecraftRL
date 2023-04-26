FROM minedojo/minedojo

RUN pip install stable_baselines3[extra]
RUN pip install gymnasium
RUN pip install pyyaml
