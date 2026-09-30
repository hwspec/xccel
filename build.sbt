val chiselVersion = "7.13.0"
val scalatestVersion = "3.2.19"

ThisBuild / scalaVersion := "2.13.18"

lazy val root = (project in file("."))
  .settings(
    name := "package",

    // Include chisel-axi-utils Scala sources directly
    Compile / unmanagedSourceDirectories +=
      baseDirectory.value / "chisel-axi-utils" / "src" / "main" / "scala",

    libraryDependencies ++= Seq(
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
  