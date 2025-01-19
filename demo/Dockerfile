FROM maven:3.8.8-eclipse-temurin-17 AS build
WORKDIR /usr/src/app

COPY pom.xml .
COPY src ./src

RUN mvn clean package -DskipTests

FROM openjdk:17-jdk-slim
WORKDIR /usr/app

COPY --from=build /usr/src/app/target/*.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
