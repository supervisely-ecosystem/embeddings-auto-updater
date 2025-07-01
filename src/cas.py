import os
import time
import warnings
from typing import List, Optional
from urllib.parse import urlparse

import numpy as np
import supervisely as sly
from clip_client import Client
from docarray.array.document import DocumentArray
from supervisely.sly_logger import logger

import src.globals as g
from src.utils import send_request, timeit, with_retries


class SlyCasClient(Client):
    def __init__(self, server: str, credential: dict = {}, **kwargs):
        """Create a Clip client object that connects to the Clip server.
        Server scheme is in the format of ``scheme://netloc:port``, where
            - scheme: one of grpc, websocket, http, grpcs, websockets, https
            - netloc: the server ip address or hostname
            - port: the public port of the server
        :param server: the server URI
        :param credential: the credential for authentication ``{'Authentication': '<token>'}``
        """
        try:
            r = urlparse(server)
            _port = r.port
            self._scheme = r.scheme
            if self._scheme == "https":
                _new_port = 443
            elif self._scheme == "http":
                _new_port = 80
        except:
            raise ValueError(f"{server} is not a valid scheme")

        _tls = False
        if self._scheme in ("grpcs", "https", "wss"):
            self._scheme = self._scheme[:-1]
            _tls = True

        if self._scheme == "ws":
            self._scheme = "websocket"  # temp fix for the core
            if credential:
                warnings.warn("Credential is not supported for websocket, please use grpc or http")

        if self._scheme in ("grpc", "http", "websocket"):
            if self._scheme == "http":
                if r.path.startswith("/net/") and _port is None:
                    _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)
                elif r.path.startswith("/net/") and _port:
                    _new_port = _port
                    _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)
            else:
                _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)

            from jina import Client

            self._client = Client(**_kwargs)
            self._async_client = Client(**_kwargs, asyncio=True)
            self._client.args.port = str(_new_port) + r.path
            self._async_client.args.port = str(_new_port) + r.path
        else:
            raise ValueError(f"{server} is not a valid scheme")

        self._authorization = credential.get("Authorization", os.environ.get("CLIP_AUTH_TOKEN"))

    @staticmethod
    def _gather_result(response, results: "DocumentArray", attribute: Optional[str] = None):
        r = response.docs
        if attribute:
            results[r[:, "id"]][:, attribute] = r[:, attribute]


class CasClient:
    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        raise NotImplementedError


class CasTaskClient(CasClient):
    # TODO This class is not used and must be refactored before to be used
    def __init__(self, api: sly.Api, task_id: int):
        self.api = api
        self.task_id = task_id

    async def get_vectors(self, queries: List[str]) -> List[List[float]]:
        return await send_request(
            self.api,
            self.task_id,
            "get_vectors",
            data={},
            context={"queries": queries},
            retries=3,
            timeout=60 * 5,
        )


class CasUrlClient(CasClient):
    STARTUP_TIMEOUT = 60 * 5  # 5 minutes

    def __init__(self, url: str):
        self.url = url
        self.client = SlyCasClient(url)
        self.__wait_for_start()

    def __wait_for_start(self):
        logger.info("Connecting to CLIP at %s...", self.url)
        t = time.monotonic()
        delay = 1
        last_exception = None
        while time.monotonic() - t < self.STARTUP_TIMEOUT:
            try:
                self.client.profile()
                logger.info("Connected to CLIP at %s!", self.url)
                return
            except Exception as e:
                last_exception = e
                logger.debug(
                    "Failed to connect to CLIP at %s. Retrying after %d seconds...",
                    self.url,
                    delay,
                    exc_info=True,
                )
                time.sleep(delay)
                if delay < 4:
                    delay *= 2
        raise RuntimeError(f"Failed to connect to CLIP at {self.url}") from last_exception

    @with_retries(retries=5, sleep_time=2)
    @timeit
    async def get_vectors(self, queries: List[str]) -> List[np.ndarray]:
        """Use CLIP to get vectors from the list of queries.
        List of queries is a list of URLs for images or text prompts.

        :param queries: List of queries (URLs for images or text prompts).
        :type queries: List[str]
        :return: List of vectors.
        :rtype: List[np.ndarray]
        """
        vectors = await self.client.aencode(queries)
        return vectors.tolist()


def _init_client() -> CasClient:
    if isinstance(g.clip_host, int):
        return CasTaskClient(
            g.api, g.clip_host
        )  # to switch on this mode you need to refactor processing of clip_host in globals.py
    else:
        return CasUrlClient(g.clip_host)


client = _init_client()


@timeit
async def get_vectors(queries: List[str]) -> List[List[float]]:
    return await client.get_vectors(queries)
