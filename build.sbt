ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "MoneyLaunderingDetection",
    version := "0.1",
    scalaVersion := "2.12.18",

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.3.2",
      "org.apache.spark" %% "spark-sql"  % "3.3.2",
      "org.apache.spark" %% "spark-mllib" % "3.3.2"
    )
  )
