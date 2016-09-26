
name := "sandbox_flooding"

organization := "com.eleuth"

version := "1.0.0"

scalaVersion := "2.11.8"

mainClass in (Compile) := Some("flat.launcher.Launcher")

scalacOptions ++= Seq("-optimize", "-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-Xlint")

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "com.eleuth" %% "seeyourmusic" % "1.0.0"
)

publishArtifact in (Compile, packageSrc) := false
