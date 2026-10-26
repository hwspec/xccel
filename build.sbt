val chiselVersion = "7.13.0"
val scalatestVersion = "3.2.19"

ThisBuild / scalaVersion := "2.13.18"

lazy val root = (project in file("."))
  .settings(
    name := "package",

    resolvers += "central-snapshots" at "https://central.sonatype.com/repository/maven-snapshots/",

    libraryDependencies ++= Seq(
      "io.github.hwspec" %% "garageworks" % "0.4.0-SNAPSHOT",
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.scalatest" %% "scalatest" % scalatestVersion % Test,

      // Equivalent of Mill's org.chipsalliance:::chisel-plugin
      compilerPlugin(
        "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
      )
    ),

    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations"
    )
  )
  