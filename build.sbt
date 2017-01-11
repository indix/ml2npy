val libVersion = sys.env.getOrElse("SNAP_PIPELINE_COUNTER", "0.1.0-SNAPSHOT")

lazy val commonSettings = Seq(
  organization := "com.indix",
  version := libVersion,
  autoAPIMappings := true,
  organizationName := "Indix",
  organizationHomepage := Some(url("http://oss.indix.com")),
  scalaVersion := "2.11.8",
  scalacOptions ++= Seq("-encoding", "UTF-8", "-deprecation", "-unchecked"),
  javacOptions ++= Seq("-Xlint:deprecation", "-source", "1.7")
)

lazy val publishSettings = Seq(
  publishMavenStyle := true,
  publishTo := {
    val nexus = "https://oss.sonatype.org/"
    if (isSnapshot.value)
      Some("snapshots" at nexus + "content/repositories/snapshots")
    else
      Some("releases" at nexus + "service/local/staging/deploy/maven2")
  },
  publishArtifact in Test := false,
  pomIncludeRepository := { _ => false },
  pomExtra := <url>https://github.com/indix/ml2npy</url>
    <licenses>
      <license>
        <name>Apache License</name>
        <url>https://raw.githubusercontent.com/indix/ml2npy/master/LICENSE</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
    <scm>
      <url>git@github.com:indix/ml2npy.git</url>
      <connection>scm:git:git@github.com:indix/ml2npy.git</connection>
    </scm>
    <developers>
      <developer>
        <id>indix</id>
        <name>Indix</name>
        <url>http://oss.indix.com</url>
      </developer>
    </developers>
)

lazy val ml2npy = (project in file(".")).
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(
    name := "ml2npy",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-mllib" % "2.0.0-preview" % "provided",
      "log4j" % "log4j" % "1.2.17" % "provided",
      "org.scalactic" %% "scalactic" % "2.2.3",
      "org.scalatest" %% "scalatest" % "2.2.3" % Test
    )
  )