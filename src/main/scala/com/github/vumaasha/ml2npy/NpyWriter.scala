package com.github.vumaasha.ml2npy

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}


/**
  * Created by vumaasha on 24/12/16.
  */
object NpyWriter {
  val magic: Array[Byte] = 0X93.toByte +: "NUMPY".getBytes(StandardCharsets.US_ASCII)
  val majorVersion = 1
  val minorVersion = 0
  val headerLenth = 70
  val dtype = "<f4"
  /*
  supported dtypes i2,i4,i8,f4,f8
   */
  val dataLength = "2"
  val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"
  val data: Array[Float] = Array(0.3f, 0.4f)

  val versionByteSize = 2
  val headerByteSize = 2
  val newLineSize = 1
  val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
  val isPaddingRequired = (unpaddedLength % 16) != 0
  val paddingLength = (((unpaddedLength / 16) + 1) * 16) - unpaddedLength
  val paddedDescription = if (isPaddingRequired) {
    description + (" " * paddingLength)
  } else description

  private val headerLength: Int = unpaddedLength + paddingLength
  val content = ByteBuffer.allocateDirect(headerLength + (data.length * 4))
    .order(ByteOrder.LITTLE_ENDIAN)
    .put(magic)
    .put(majorVersion.toByte)
    .put(minorVersion.toByte)
    .putShort(paddedDescription.length.toShort)
    .put(paddedDescription.getBytes(StandardCharsets.US_ASCII))

  data.foreach(x => content.putFloat(x))
  content.flip

  def main(args: Array[String]): Unit = {
    val channel = FileChannel.open(Paths.get("/tmp/test.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    channel.write(content)
    channel.truncate(content.position)
    channel.close()
  }
}
