FROM jenkins/jenkins:lts
USER root

RUN apt-get update && apt-get install -y openjdk-11-jdk
RUN apt-get install -y git

# Устанавливаем python3, pip, venv если они еще не установлены
RUN apt-get install -y python3 python3-pip python3-venv

# создаем виртуальное окружение
RUN python3 -m venv /opt/venv

# активируем виртуальное окружение
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
	pip install numpy pandas scikit-learn 

# возвращаем пользователя jenkins
USER jenkins

# Открываем порт 8080 для доступа к Jenkins
#EXPOSE 8080

# Запускаем Jenkins в фоновом режиме
#CMD ["jenkins", "-arg", "--httpPort=8080"]
