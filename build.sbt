name := "spark-laboratory"

version := "0.0.1-SNAPSHOT"

scalaVersion := "2.11.8"

val sparkLibVersion= "2.0.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkLibVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkLibVersion withSources()
)
