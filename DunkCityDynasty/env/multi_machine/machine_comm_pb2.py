# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: machine_comm.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12machine_comm.proto\x12\x04\x63omm\"Y\n\tClientCmd\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\x0b\n\x03\x63md\x18\x02 \x01(\t\x12\x14\n\x0crl_server_ip\x18\x03 \x01(\t\x12\x16\n\x0erl_server_port\x18\x04 \x01(\x05\"\x14\n\x05Reply\x12\x0b\n\x03msg\x18\x01 \x01(\t23\n\nClientComm\x12%\n\x03\x43md\x12\x0f.comm.ClientCmd\x1a\x0b.comm.Reply\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'machine_comm_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CLIENTCMD._serialized_start=28
  _CLIENTCMD._serialized_end=117
  _REPLY._serialized_start=119
  _REPLY._serialized_end=139
  _CLIENTCOMM._serialized_start=141
  _CLIENTCOMM._serialized_end=192
# @@protoc_insertion_point(module_scope)
