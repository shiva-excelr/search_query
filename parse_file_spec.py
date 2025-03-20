import yaml
import json
from datetime import date, datetime
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
import requests
from extractors.utils import get_current_iso_time


class OpenAPISpecProcessor:
    def __init__(self, spec_data):

        # self.spec = yaml.safe_load(spec_data.decode('utf-8'))
        self.spec =spec_data

        self.components = self.spec.get("components", {})

    def resolve_reference(self,ref_path):
        """Resolve $ref from components."""
        keys = ref_path.replace("#/", "").split("/")
        ref_value = self.components
        for key in keys[1:]:
            ref_value = ref_value.get(key, {})
        return ref_value


    def extract_security_details(self, global_security,path_details):
        """
        Extracts security schemes and consolidates their details, excluding those with 'in: header'.

        Args:
            openapi_spec_path (str): Path to the OpenAPI specification file.

        Returns:
            list: A list of dictionaries containing authType, required, and value fields for all security schemes.
        """


        security_schemes = self.components.get("securitySchemes", {})


        # Consolidate unique security schemes and exclude 'in: header'
        unique_schemes = {}
        for name, details in security_schemes.items():
            if details.get("in") != "header":  # Exclude schemes with 'in: header'
                unique_schemes[name] = details

        for security in global_security:
            for name in security:
                if name not in security_schemes:
                    unique_schemes[name] = {"type": "unknown", "description": "Defined in global security"}

        for security in path_details.get('security',[]):
            for name in security:
                if name not in security_schemes:
                    unique_schemes[name] = {"type": "unknown", "description": "Defined in global security"}
        # Create the security details list
        security_details = {}
        for name, details in unique_schemes.items():
            auth_type = details.get("type", "unknown")
            basic_config = {"authActive": True}
            if auth_type == "http":
                scheme = details.get("scheme")
                if scheme == "basic":
                   basic_config.update({
                       "authType": "basic",
                       "password": "",
                       "user": ""
                   })
                elif scheme == "bearer":
                    basic_config.update({
                        "authType": "bearer",
                        "token": "",
                    })
                basic_config['authType'] = scheme
                security_details.update(**basic_config)
            elif auth_type == "oauth2":
                all_flows = details.get("flows",{})
                if len(all_flows) > 1:
                    auth_flows = self.determine_oauth_flows(all_flows)
                    return auth_flows
                else:
                    oauth = self.determine_oauth_flows(all_flows)[0]
                    security_details.update(**oauth)

            elif auth_type == "apiKey":
                basic_config.update({
                    "authType": "api-key",
                    "passBy": "",
                    "key": "",
                    "value": ""
                })
                security_details.update(**basic_config)




        return security_details if  security_details else {"authType": "none"}

    def determine_oauth_flows(self, flows):
        token_url = auth_url = ''
        auth_flows = []
        scopes = ''
        for flow in flows:
            if flow == "authorizationCode":
                auth_url = flows[flow].get("authorizationUrl",'')
                token_url = flows[flow].get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow =="implicit":
                auth_url = flows[flow].get("authorizationUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow == "clientCredentials":
                token_url = flows[flow].get("clientCredentials", {}).get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow == "password":
                token_url = flows[flow].get("password", {}).get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])

            auth_flows.append( {
                "authType": "oauth-2",
                "accessTokenURL": token_url,
                "authURL": auth_url,
                "clientId": "",
                "clientSecret": "",
                "discoveryURL": "",
                "scope": scopes,
                "token": ""
            })
        return auth_flows

    def extract_path_parameters_from_openapi(self, parameters, path_item, path) -> List[Dict[str, Any]]:
        """
        Extracts path parameters from an OpenAPI specification.

        :param openapi_spec: Parsed OpenAPI spec as a dictionary.
        :return: A list of path parameter details.
        """
        path_parameters = []


        if "parameters" in path_item:
            for param in path_item["parameters"]:
                if "$ref" in param:
                    param = self.resolve_reference(param["$ref"])
                if param.get("in") == "path":
                    path_parameters.append({ **param})

        # Extract parameters defined at the method level


        for param in parameters:
            if "$ref" in param:
                param = self.resolve_reference(param["$ref"])
            if param.get("in") == "path":
                if 'schema' in param and isinstance(param['schema'],set):
                    param['schema'] = str(param['schema'])
                path_parameters.append({ **param})

        # Resolve implicit parameters from the path string
        # for path, path_item in openapi_spec.get("paths", {}).items():
        implicit_params = [segment.strip("{}").strip() for segment in path.split("/") if
                           segment.startswith("{") and segment.endswith("}")]
        for param_name in implicit_params:
            # Check if the parameter is already defined
            #and param.get("path") == path
            if not any(param.get("name") == param_name  for param in path_parameters):
                path_parameters.append({
                    "path": path,
                    "name": param_name,
                    "required": True})

        return path_parameters

    def extract_headers_from_openapi(self,parameters,path_item) ->List[Dict[str, Any]]:
        """
        Extracts headers from an OpenAPI specification.

        :param openapi_spec: Parsed OpenAPI spec as a dictionary.
        :return: A dictionary with header details grouped by type.
        """

        headers = {'headers':[]}

        security_names = [scheme_name for scheme_name, scheme in self.components["securitySchemes"].items()]

        # Extract explicit request headers (parameters in: header)

        if "parameters" in path_item:
            for param in path_item["parameters"]:
                if "$ref" in param:
                    param = self.resolve_reference(param["$ref"])
                if param.get("in") == "header" and param.get("name") not in security_names:
                    required = param.get("required", False)
                    headers["headers"].append({"enabled": required, **param})


        for param in parameters:
            required = param.get("required", False)
            if "$ref" in param:
                param = self.resolve_reference(param["$ref"])
            if param.get("in") == "header" and param.get("name") not in security_names:
                headers["headers"].append({"enabled": required, **param})

        # Extract security scheme headers (e.g., Authorization)
        if  "securitySchemes" in self.components:
            for scheme_name, scheme in self.components["securitySchemes"].items():
                if "$ref" in scheme:
                    scheme = self.resolve_reference(scheme["$ref"])
                if scheme.get("in") == "header":
                    required = scheme.get("required", False)
                    headers["headers"].append({"enabled": required, **scheme})

        # Extract implicit headers from content negotiation (Content-Type, Accept)
        # if "paths" in openapi_spec:
        #     for path, path_item in openapi_spec["paths"].items():
        #         for method, operation in path_item.items():
        #             if method in ["get", "post", "put", "delete", "patch", "options", "head"]:
        #                 if "requestBody" in operation:
        #                     if "$ref" in operation["requestBody"]:
        #                         operation["requestBody"] = self.resolve_reference(operation["requestBody"]["$ref"],
        #                                                                      self.components)
        #                     if "content" in operation["requestBody"]:
        #                         required = operation["requestBody"].get("required", False)
        #                         ##TODO check the details in implicit_headers for recursion values
        #                         headers["implicit_headers"].append({
        #                             "path": path,
        #                             "method": method,
        #                             "enabled": required,
        #                             "header": "Content-Type",
        #                             "details": list(operation["requestBody"]["content"].keys())
        #                         })
        # Extract headers defined in extensions (x-*)
        # for key, value in openapi_spec.items():
        #     if key.startswith("x-") and isinstance(value, dict):
        #         headers["extensions"].append(value)

        return headers['headers']

    def extract_query_parameters(self,parameters):


        def get_value(param):
            """Extract value from parameter details."""
            if "example" in param:
                return param["example"] if not isinstance(param["example"],dict) else ''
            if "schema" in param:
                schema = param["schema"]
                if "default" in schema:
                    return schema["default"]
                elif "example" in schema:
                    return schema["example"] if not isinstance(schema["example"],dict) else schema["example"].get("eq",'')
                elif "items" in schema and "$ref" in schema["items"]:
                    ref_value = self.resolve_reference(schema["items"]["$ref"])
                    return ref_value.get("default") or ref_value.get("example") or '|'.join(ref_value.get("enum",''))
                elif "items" in schema and schema.get("type") == "array":
                    values = [f'string{i+1}' for i in range(2)]
                    return  values
            if "$ref" in param:
                ref_value = self.resolve_reference(param["$ref"])
                return get_value(ref_value)
            return ''

        query_parameters = []
        components = self.components


        for param in parameters:
            if param.get("in") == 'query':
                resolved_param = param
                if "$ref" in param:
                    resolved_param = self.resolve_reference(param["$ref"])
                name = resolved_param["name"]
                required = resolved_param.get("required", False)
                value = get_value(resolved_param)
                if isinstance(value,list):
                    for val in value:
                        query_parameters.append({"name": name, "value": str(val), "enabled": required})
                else:
                   query_parameters.append({"name": name, "value": str(value), "enabled": required})

        return query_parameters




    # def resolve_ref(self, ref):
    #     """Resolves a $ref string to the corresponding component."""
    #     if not ref.startswith("#/components/"):
    #         raise ValueError(f"Unsupported reference format: {ref}")
    #     path = ref.lstrip("#/").split("/")
    #     ref_value = self.components
    #     for key in path[1:]:
    #         ref_value = ref_value.get(key, {})
    #     return ref_value

    def resolve_recursive(self, obj, key='', note='starting', resolved_refs=None):
        """Recursively resolves references in a dictionary, handling nested and recursive structures."""
        if resolved_refs is None:
            resolved_refs = {}

        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref in resolved_refs:
                    return resolved_refs[ref]
                resolved = self.resolve_reference(ref)
                resolved_refs[ref] = resolved
                # Recursively resolve the resolved object
                return self.resolve_recursive(resolved, key="ref", note="ref instance", resolved_refs=resolved_refs)
            else:
                return {k: self.resolve_recursive(v, k, note="dict instance", resolved_refs=resolved_refs) for k, v in
                        obj.items()}
        elif isinstance(obj, list):
            return [self.resolve_recursive(item, key, note="list instance", resolved_refs=resolved_refs) for item in
                    obj]
        else:
            return obj

    def resolve_recursive_iterative(self, obj):
        stack = [(None, obj)]  # Stack to track parent and current object
        result = {}

        while stack:
            parent, current = stack.pop()

            if isinstance(current, dict):
                if "$ref" in current:
                    resolved = self.resolve_reference(current["$ref"])
                    stack.append((parent, resolved))
                else:
                    resolved_dict = {}
                    for k, v in current.items():
                        stack.append((resolved_dict, (k, v)))
                    if parent is None:
                        result = resolved_dict
                    else:
                        parent.append(resolved_dict)
            elif isinstance(current, list):
                resolved_list = []
                for item in current:
                    stack.append((resolved_list, item))
                if parent is None:
                    result = resolved_list
                else:
                    parent.append(resolved_list)
            else:
                if isinstance(parent, dict):
                    k, v = current
                    parent[k] = v
                elif isinstance(parent, list):
                    parent.append(current)

        return result

    def extract_request_details(self,request_info = {}):
        """Extracts all HTTP request details from the spec."""
        paths = self.spec.get("paths", {})
        request_collection = []
        servers = self.spec.get("servers", [])
        global_security  =self.spec.get("security", [])
        collection_name = self.spec.get("info",{}).get("title") if not request_info.get("collection_name") else request_info.get("collection_name")

        for path, methods in paths.items():
            for method, details in methods.items():

                if method not in ["get", "post", "put", "delete", "patch", "options", "head"]:
                    continue

                #TODO Parameters in methods to extract as common headers

                request_info = {
                    "servers": servers,
                    "path": path,
                    "method": method,
                    "summary": details.get("summary", details.get("operationId",path)),
                    "description": details.get("description", ""),
                    "headers": {},
                    "query_params": {},
                    "body": None,
                    "path_params": {},
                }

                # Extract parameters
                parameters = details.get("parameters", methods.get("parameters",[]))
                request_info['query_params'] = self.extract_query_parameters(parameters)
                request_info['headers'] = self.extract_headers_from_openapi(parameters,methods)
                request_info['path_params'] = self.extract_path_parameters_from_openapi(parameters,methods,path, )
                request_info['auth'] = self.extract_security_details(global_security,details)
                request_info['collection_name'] = collection_name



                # Extract request body
                if "requestBody" in details:
                    resolved_body = self.resolve_recursive(details["requestBody"])
                    request_info["body"] = resolved_body

                if "responses" in details:
                    try:
                        success_code =  [item for item in details["responses"] if  item.isdigit() and (200 <=  int(item) < 300)]
                        if len(success_code) > 0:
                            resolved_body = self.resolve_recursive(details["responses"][success_code[0]])
                            request_info["response"] = resolved_body
                        elif "default" in details["responses"]:
                            resolved_body = self.resolve_recursive(details["responses"]['default'])
                            request_info["response"] = resolved_body
                        else:
                            request_info["response"] = {}
                    except ValueError as e :
                        if 'default' in details["responses"]:
                            resolved_body = self.resolve_recursive(details["responses"]['default']['content']['application/json'])
                            request_info["response"] = resolved_body
                    except Exception as e:
                        request_info["response"] = {}

                request_collection.append(request_info)

        return request_collection

    def save_collection(self, collection, output_file):
        """Saves the request collection to a JSON file."""
        def custom_serializer(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(output_file, 'w') as file:
            json.dump(collection, file, indent=4, default=custom_serializer)

    def get_body(self, schema):
        def extract_properties(schema):
            """Extract properties from schema, considering all possible locations."""
            properties = {}

            # If 'properties' is directly in schema
            if 'properties' in schema:
                properties.update(schema['properties'])

            # If 'items' contains 'properties'
            if 'items' in schema and 'properties' in schema['items']:
                properties.update(schema['items']['properties'])

            # If 'allOf' is in schema, merge properties from each element
            if 'allOf' in schema:
                for sub_schema in schema['allOf']:
                    properties.update(extract_properties(sub_schema))

            return properties

        def extract_required(schema):
            """Extract required fields from schema."""
            required = schema.get("required", [])

            # If 'allOf' is in schema, combine required fields from each element
            if 'allOf' in schema:
                for sub_schema in schema['allOf']:
                    required += extract_required(sub_schema)

            return list(set(required))  # Remove duplicates

        # Extract properties and required fields
        properties = extract_properties(schema)
        required = extract_required(schema)
        data_type = schema.get("type","string")

        # Rearrange properties: required first, then others
        rearranged = {key: properties[key] for key in required if key in properties}
        rearranged.update({key: properties[key] for key in properties if key not in required})

        final_body = {}

        for data, key in rearranged.items():
            if key.get("allOf"):
                body_of_allof = {}
                for sub_schema in key["allOf"]:

                    result = self.extract_key_types(data, {data: sub_schema})
                    newkey, value = next(iter(result.items()))

                    if not isinstance(value,dict):
                        body_of_allof = value
                    else:
                        body_of_allof.update(**value)

                    # body_of_allof.update(self.get_body(sub_schema))

                final_body[data] = body_of_allof
            elif key.get("properties"):
                result = self.extract_key_types(data, {data: key})
                key, value = next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = self.get_body(key)

            elif key.get("items"):
                result = self.extract_key_types(data, {data: key})
                key, value = next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = self.get_body(key)
            else:
                result = self.extract_key_types(data, {data:key})
                key, value =  next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = key.get("example", f"{key.get('description','')} {key.get('format', '')} {key['type']}")


        return [final_body] if data_type =="array" else final_body



    def resolve_ref_url(self, ref, spec):
        """Resolves a $ref within the OpenAPI spec."""
        if not ref.startswith("#/"):
            raise ValueError(f"Unsupported $ref format: {ref}")

        path = ref.lstrip("#/").split("/")
        ref_data = spec
        for part in path:
            ref_data = ref_data.get(part, {})
        return ref_data

    def extract_and_replace_urls(self,urls,servers):
        final_urls = []
        for url,server in zip(urls,servers):
            region_info = server.get('variables', {}).get('region', {})
            region = region_info.get('default') or region_info.get('enum', [])[0] if len(region_info.get('enum', [])) > 0 else []
            if region:
                final_urls.append(url.replace('{region}',region))
            else:
                final_urls.append(url)
        return final_urls

    def extract_server_urls(self,openapi_spec):
        # Default URL if no servers are found
        default_url = "http://localhost:0001"
        urls = []

        # Check if `servers` section is present
        if 'servers' in openapi_spec:
            for server in openapi_spec['servers']:
                if '$ref' in server:  # Resolve $ref
                    ref_data = self.resolve_ref_url(server['$ref'], openapi_spec)
                    if 'url' in ref_data:
                        urls.append(ref_data['url'])
                elif 'url' in server:
                    urls.append(server['url'])

        # Check if `components` section has server-related definitions
        if 'components' in openapi_spec:
            components = openapi_spec['components']
            for key, value in components.items():
                if isinstance(value, dict) and 'properties' in value:
                    props = value['properties']
                    if 'url' in props and isinstance(props['url'], dict):
                        urls.append(props['url'].get('default', default_url))

        # Return the URLs found or the default if none
        urls = self.extract_and_replace_urls(urls,openapi_spec['servers'])
        return urls if urls else [default_url]


    def dict_to_xml(self,tag, schema):
        elem = ET.Element(tag)

        for key, value in schema.items():
            if isinstance(value, dict):
                if "properties" in value:
                    # Nested object
                    child = self.dict_to_xml(value.get("xml", {}).get("name", key), value["properties"])
                    elem.append(child)
                elif "example" in value:
                    # Leaf node with an example
                    child = ET.Element(key)
                    child.text = str(value["example"])
                    elem.append(child)
                elif "items" in value:
                    # Handle array
                    array_tag = key
                    wrapped = value.get("xml", {}).get("wrapped", False)

                    if wrapped:
                        wrapper = ET.Element(array_tag)
                        for item in value["items"].get("xml", []):
                            if "properties" in value["items"]:
                                child = self.dict_to_xml(value["items"].get("xml", {}).get("name", key), value["items"]["properties"])
                                elem.append(child)
                            else:
                                item_tag = value["items"].get("xml", {}).get("name", "item")
                                item_elem = ET.Element(item_tag)
                                item_elem.text = str(item)
                                wrapper.append(item_elem)
                            elem.append(wrapper)
                    else:
                        for item in value["items"].get("example", []):
                            item_elem = ET.Element(array_tag)
                            item_elem.text = str(item)
                            elem.append(item_elem)

            elif isinstance(value, list):
                # Handle list of primitive values
                for item in value:
                    child = ET.Element(key)
                    child.text = str(item)
                    elem.append(child)

        return elem

    def generate_xml_body(self,spec: Dict[str, Any]) -> str:
        type_defaults = {
            "string": "string",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {}
        }

        def create_element(name: str, properties: dict, parent: ET.Element) -> None:
            # Handle wrapped arrays
            if properties.get('type') == 'array' and properties.get('xml', {}).get('wrapped'):
                wrapper = ET.SubElement(parent, name)
                item_props = properties['items']
                item_name = item_props.get('xml', {}).get('name', 'item')

                # Create example array item
                if 'properties' in item_props:
                    sub_elem = ET.SubElement(wrapper, item_name)
                    for prop_name, prop_value in item_props['properties'].items():
                        create_element(prop_name, prop_value, sub_elem)
                else:
                    sub_elem = ET.SubElement(wrapper, item_name)
                    sub_elem.text = item_props.get('example', 'string')
                return

            # Handle regular properties
            element = ET.SubElement(parent, name)
            if properties.get('type') == 'object' and 'properties' in properties:
                for prop_name, prop_value in properties['properties'].items():
                    create_element(prop_name, prop_value, element)
            else:
                if 'example' in properties:
                    element.text = str(properties['example'])
                elif 'enum' in properties:
                    element.text = str(properties['enum'][0])
                elif 'type' in properties:
                    element.text = str(type_defaults[properties['type']])
                else:
                    element.text = ''

        # Create root element
        root_name = spec.get('xml', {}).get('name', 'root')
        root = ET.Element(root_name)

        # Process all properties
        for prop_name, prop_value in spec['properties'].items():
            create_element(prop_name, prop_value, root)

        # Convert to string with pretty printing
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml()

        # Add XML declaration
        if not xml_str.startswith('<?xml'):
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

        return xml_str



    def convert_to_form_data(self,data):

        form_data = []

        for key, value in data.items():

            form_data.append({
                "key": key,
                "value": json.dumps(value) if isinstance(value, dict) or isinstance(value, list) else str(value),
                "active": True
            })

        return form_data

    def convert_into_prutan_spec(self,data):

        specs = []

        for request in data:
            servers = self.extract_server_urls(request)
            if not servers:
                print("No server url's present")
                return specs


            base_spec = {
                'collection_name':request['collection_name'],
                'method': request['method'].upper(),
                'endpoint': servers[0] + request["path"],
                'request_name':  request['method'].upper() + " "+request["summary"],
                'response_content': []
            }
            path_params = request['path_params'] if request['path_params']  else []

            if request.get('response'):
                response_content_types = []
                for content in request['response'].get('content', []):
                    response_content_types.append(content)
                base_spec.update({"response_content":response_content_types})
                    # if content == "application/json":
                    #     if isinstance(request['response'].get('content', {}).get(content, {}), dict):
                    #         response = self.get_body(request['response'].get('content', {}).get(content, {})['schema'])
                    #     elif isinstance(request['response'].get('content', {}).get(content, {}), set):
                    #         response= str(request['response'].get('content', {}).get(content))
                    #     else:
                    #        response = request['response'].get('content', {}).get(content)
                    #     base_spec["responseContent"] = json.dumps(response) if isinstance(response,(dict, list)) else str(response)
                    #     break

            if base_spec["response_content"] == []:
                base_spec["response_content"] = ['application/json']

            # Handle body content if it exists
            if request.get('body'):
                for content in request['body'].get('content', []):
                    spec = base_spec.copy()
                    for accept in spec['response_content']:
                        header_copy = request.get('headers', []).copy()
                        spec["content_type"] = content
                        if content == "application/json":
                            if isinstance(request['body'].get('content',{}).get(content,{}), dict):
                              spec["body"] = self.get_body(request['body'].get('content',{}).get(content,{})['schema'])
                            elif isinstance(request['body'].get('content',{}).get(content,{}), set):
                                spec["body"] = str(request['body'].get('content',{}).get(content))
                            else:
                                spec["body"] = request['body'].get('content',{}).get(content)
                        elif content == "application/xml":
                            body_contents = request['body'].get('content',{}).get(content,{})['schema']
                            xml_request_body = self.generate_xml_body(body_contents)
                            spec["body"] = xml_request_body
                        elif content == "application/x-www-form-urlencoded":
                            form_body = self.get_body(request['body'].get('content',{}).get(content,{})['schema'])
                            spec['body'] = self.convert_to_form_data(form_body)

                        elif content == "multipart/form-data":
                            form_body = self.get_body(request['body'].get('content', {}).get(content, {})['schema'])
                            spec['body'] = self.convert_to_form_data(form_body)

                        elif content == "application/octet-stream":
                            form_body = self.get_body(request['body'].get('content', {}).get(content, {})['schema'])
                            spec['body'] = form_body

                        header_copy.append({
                            "name": "accept",
                            "value": accept,
                            "enabled": True
                        })

                        if header_copy:
                            spec["headers"] = header_copy

                        if request.get('query_params'):
                            spec["params"] = request['query_params']
                            url = base_spec['endpoint']
                            for param in request['query_params']:
                                url += f"?{param.get("name")}={param.get("value", '')}"
                            base_spec['endpoint'] = url
                        if request.get('path_params'):
                            spec["path_params"] = request['path_params']
                        if request.get('auth'):
                            spec["auth"] = request['auth']

                        spec_copy = spec.copy()
                        specs.append(spec_copy)
            else:
                for accept in base_spec["response_content"]:
                    header_copy = request.get('headers', []).copy()
                    header_copy.append({
                        "name": "accept",
                        "value": accept,
                        "enabled": True
                    })
                    if header_copy:
                        base_spec["headers"] = header_copy

                    if request.get('query_params'):
                        base_spec["params"] = request['query_params']
                        url = base_spec['endpoint']
                        for param in request['query_params']:
                            url += f"?{param.get("name")}={param.get("value", '')}"
                        base_spec['endpoint'] = url

                    if request.get('path_params'):
                        base_spec["path_params"] = request['path_params']

                    spec_copy = base_spec.copy()
                    specs.append(spec_copy)

        return specs

    def process_schema(self,schema,prop=''):
        """
        Process the schema and return its value based on the type and other conditions.
        """

        if not isinstance(schema,dict):
            return schema
        if "example" in schema:
            if isinstance(schema['example'],datetime) or isinstance(schema['example'],date):
                return str(schema["example"])
            return schema["example"]

        if "examples" in schema:
            # If examples are provided, return an empty list of the correct type
            return []

        if "enum" in schema:
            value = schema["enum"][0] if schema["enum"] else None
            return value

        if "nullable" in schema and schema["nullable"]:
            return None

        if schema.get("type") == "object":
            result = {}
            if schema.get("properties") is not None:
                for prop, prop_schema in schema.get("properties", {}).items():
                    result[prop] = self.process_schema(prop_schema,prop)
            if "additionalProperties" in schema:
                additional_schema = schema["additionalProperties"]
                for i in range(3):
                   result[f"additionalProp{i+1}"] = self.process_schema(additional_schema)
                # result["additionalProperty"] = self.process_schema(additional_schema)
            return result

        if schema.get("type") == "array":
            items_schema = schema.get("items", {})
            return [self.process_schema(items_schema)]

        if "allOf" in schema:
            result = {}
            for sub_schema in schema["allOf"]:
                val = self.process_schema(sub_schema)
                if not isinstance(val,dict):
                    result = val
                else:
                   result.update(val)
            return result

        if "oneOf" in schema or "anyOf" in schema:
            options = schema.get("oneOf") or schema.get("anyOf")
            return [self.process_schema(option) for option in options]

        if "discriminator" in schema:
            # Handle discriminator if necessary (usually requires additional logic)
            return {"discriminator": schema["discriminator"]}

        # Return a default value based on the type
        type_defaults = {
            "string": "string",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {}
        }
        if schema and 'date' in schema.get("format",{}):
            return get_current_iso_time()
        return type_defaults.get(schema.get("type"), type_defaults.get("string"))


    def extract_key_types(self,key, value):
        """
        Extract key types and their values from a schema.
        """
        result = {}

        value = value[key]

        if value.get("type") == "array":
            item_type = self.process_schema(value.get("items", {}))
            result[key] = [item_type]
        elif value.get("type") == "object":
            result[key] = self.process_schema(value)
        elif "enum" in value:
            result[key] = value["enum"][0] if value["enum"] else None
        else:
            result[key] = self.process_schema(value)

        return result




class SwaggerSpecProcessor:
    def __init__(self, spec_data):

        self.spec =spec_data

        self.components = self.spec.get("components", {})

    def resolve_reference(self,ref_path):
        """Resolve $ref from components."""
        keys = ref_path.replace("#/", "").split("/")
        ref_value = self.spec
        for key in keys:
            ref_value = ref_value.get(key, {})
        return ref_value


    def extract_security_details(self, global_security,path_details):
        """
        Extracts security schemes and consolidates their details, excluding those with 'in: header'.

        Args:
            openapi_spec_path (str): Path to the OpenAPI specification file.

        Returns:
            list: A list of dictionaries containing authType, required, and value fields for all security schemes.
        """


        security_schemes = self.components.get("securitySchemes", {})

        security_definitions = self.spec.get("securityDefinitions",{})

        security_names = [security_definitions[i].get("name",i) for i in security_definitions if
                          security_definitions[i].get("name",i)]




        # Consolidate unique security schemes and exclude 'in: header'
        unique_schemes = {}
        for name, details in security_definitions.items():
            if details.get("in") not in ["header","query"]:  # Exclude schemes with 'in: header'
                unique_schemes[name] = details

        for security in global_security:
            for name in security:
                if name not in security_names:
                    unique_schemes[name] = {"type": "unknown", "description": "Defined in global security"}

        for security in path_details.get('security',[]):
            for name in security:
                if name not in security_definitions:
                    unique_schemes[name] = {"type": "unknown", "description": "Defined in global security"}
        # Create the security details list
        security_details = {}
        for name, details in unique_schemes.items():
            auth_type = details.get("type", "unknown")
            basic_config = {"authActive": True}

            if auth_type == "basic":
               basic_config.update({
                   "authType": "basic",
                   "password": "",
                   "user": ""
               })
               security_details.update(**basic_config)
            elif auth_type == "bearer":
                basic_config.update({
                    "authType": "bearer",
                    "token": "",
                })
                security_details.update(**basic_config)
            elif auth_type == "oauth2":
                all_flows = details
                # if len(all_flows) > 1:
                #     auth_flows = self.determine_oauth_flows(all_flows)
                #     return auth_flows
                # else:

                # Adjust for one or more auth types
                oauth = self.determine_oauth_flows({all_flows.get("flow"):all_flows})[0]
                security_details.update(**oauth)

            elif auth_type == "apiKey":
                basic_config.update({
                    "authType": "api-key",
                    "passBy": "",
                    "key": "",
                    "value": ""
                })
                security_details.update(**basic_config)




        return security_details if  security_details else {"authType": "none"}

    def determine_oauth_flows(self, flows):
        token_url = auth_url = ''
        auth_flows = []
        scopes = ''
        for flow in flows:
            if flow == "authorizationCode":
                auth_url = flows[flow].get("authorizationUrl",'')
                token_url = flows[flow].get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow =="implicit":
                auth_url = flows[flow].get("authorizationUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow == "clientCredentials":
                token_url = flows[flow].get("clientCredentials", {}).get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])
            elif flow == "password":
                token_url = flows[flow].get("password", {}).get("tokenUrl",'')
                scopes = " ".join([i for i in flows[flow].get("scopes", '')])

            auth_flows.append( {
                "authType": "oauth-2",
                "accessTokenURL": token_url,
                "authURL": auth_url,
                "clientId": "",
                "clientSecret": "",
                "discoveryURL": "",
                "scope": scopes,
                "token": ""
            })
        return auth_flows

    def extract_path_parameters_from_swagger(self, parameters, path_item, path) -> List[Dict[str, Any]]:
        """
        Extracts path parameters from an OpenAPI specification.

        :param openapi_spec: Parsed OpenAPI spec as a dictionary.
        :return: A list of path parameter details.
        """
        path_parameters = []


        if "parameters" in path_item:
            for param in path_item["parameters"]:
                if "$ref" in param:
                    param = self.resolve_reference(param["$ref"])
                if param.get("in") == "path":
                    if 'schema' in param and isinstance(param['schema'], set):
                        param['schema'] = str(param['schema'])
                    path_parameters.append({ **param})

        # Extract parameters defined at the method level


        for param in parameters:
            if "$ref" in param:
                param = self.resolve_reference(param["$ref"])
            if param.get("in") == "path":
                # path_parameters.append({
                #     "path": path,
                #     "name": param.get("name"),
                #     "required": True})
                path_parameters.append({ **param})

        # Resolve implicit parameters from the path string
        # for path, path_item in openapi_spec.get("paths", {}).items():
        implicit_params = [segment.strip("{}").strip() for segment in path.split("/") if
                           segment.startswith("{") and segment.endswith("}")]
        for param_name in implicit_params:
            # Check if the parameter is already defined
            #and param.get("path") == path
            if not any(param.get("name") == param_name  for param in path_parameters):
                path_parameters.append({
                    "path": path,
                    "name": param_name,
                    "required": True})

                # path_parameters.append({**param})

        return path_parameters

    def extract_headers_from_swagger(self,parameters,path_item,global_security) ->List[Dict[str, Any]]:
        """
        Extracts headers from an OpenAPI specification.

        :param openapi_spec: Parsed OpenAPI spec as a dictionary.
        :return: A dictionary with header details grouped by type.
        """

        headers = {'headers':[]}
        security_definitions = self.spec.get("securityDefinitions",[])

        security_names = [security_definitions[i].get("name") for i in security_definitions if security_definitions[i].get("name")]

        # Extract explicit request headers (parameters in: header)

        if "parameters" in path_item:
            for param in path_item["parameters"]:
                if "$ref" in param:
                    param = self.resolve_reference(param["$ref"])
                if param.get("in") == "header" and param.get("name") not in security_names:
                    required = param.get("required", False)
                    headers["headers"].append({"enabled": required, **param})


        for param in parameters:
            required = param.get("required", False)
            if "$ref" in param:
                param = self.resolve_reference(param["$ref"])
            if param.get("in") == "header" and param.get("name") not in security_names:
                security_def = {key: value for key, value in param.items() if  key != "description"}
                headers["headers"].append({"enabled": required, **security_def})



        for scheme_name in security_definitions:
                if security_definitions[scheme_name].get("in") == "header":
                    required = True
                    security_def = {key: value for key, value in security_definitions[scheme_name].items() if key != "description"}
                    headers["headers"].append({"enabled": required, **security_def})


        # Updating headers to avoid duplicates


        return headers['headers']

    def extract_query_parameters_from_swagger(self,parameters,global_security):


        def get_value(param):
            """Extract value from parameter details."""
            if "example" in param:
                return param["example"] if not isinstance(param["example"],dict) else ''
            if "default" in param:
                return param["default"] if not isinstance(param["default"],dict) else ''
            if "schema" in param:
                schema = param["schema"]
                if "default" in schema:
                    return schema["default"]
                elif "example" in schema:
                    return schema["example"] if not isinstance(schema["example"],dict) else schema["example"].get("eq",'')
                elif "items" in schema and "$ref" in schema["items"]:
                    ref_value = self.resolve_reference(schema["items"]["$ref"])
                    return ref_value.get("default") or ref_value.get("example") or '|'.join(ref_value.get("enum",''))
                elif "items" in schema and schema.get("type") == "array":
                    values = [f'string{i+1}' for i in range(2)]
                    return  values
            if "$ref" in param:
                ref_value = self.resolve_reference(param["$ref"])
                return get_value(ref_value)
            elif "items" in param and param.get("type") == "array":
                if "default" in param.get("items"):
                    return param.get("items",{}).get("default")
                else:
                   values = [f'string{i+1}' for i in range(2)]
                   return values
            return ''

        query_parameters = []

        security_definitions = self.spec.get("securityDefinitions", [])





        for param in parameters:
            if param.get("in") == 'query':
                resolved_param = param
                if "$ref" in param:
                    resolved_param = self.resolve_reference(param["$ref"])
                name = resolved_param["name"]
                required = resolved_param.get("required", False)
                value = get_value(resolved_param)
                if isinstance(value,list):
                    for val in value:
                        query_parameters.append({"name": name, "value": str(val), "enabled": required})
                else:
                   query_parameters.append({"name": name, "value": str(value), "enabled": required})
            else:
                if "$ref" in param:
                    resolved_param = self.resolve_reference(param["$ref"])
                    if resolved_param.get("in") == "query":
                        name = resolved_param["name"]
                        required = resolved_param.get("required", False)
                        value = get_value(resolved_param)
                        if isinstance(value, list):
                            for val in value:
                                query_parameters.append({"name": name, "value": str(val), "enabled": required})
                        else:
                            query_parameters.append({"name": name, "value": str(value), "enabled": required})

        for scheme_name in security_definitions:
            if security_definitions[scheme_name].get("in") == "query":
                required = True
                name = security_definitions[scheme_name]['name']
                value = security_definitions[scheme_name].get("value",'')
                query_parameters.append({"name": name, "value":value , "enabled": required})

        return query_parameters






    def resolve_recursive(self, obj):
        """Recursively resolves references in a dictionary."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                resolved = self.resolve_reference(obj["$ref"])
                return self.resolve_recursive(resolved)
            else:
                return {k: self.resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.resolve_recursive(item) for item in obj]
        else:
            return obj

    def extract_request_details(self,request_info = {}):
        """Extracts all HTTP request details from the spec."""
        paths = self.spec.get("paths", {})
        request_collection = []

        global_security  =self.spec.get("security", [])
        collection_name = self.spec.get("info",{}).get("title") if not request_info.get("collection_name") else request_info.get("collection_name")

        servers = self.extract_server_urls(self.spec.get("schemes", []), self.spec.get("host", ''),self.spec.get("basePath", ''))

        for path, methods in paths.items():
            for method, details in methods.items():
                if method not in ["get", "post", "put", "delete", "patch", "options", "head"]:
                    continue

                #TODO Parameters in methods to extract as common headers

                request_info = {
                    "servers": servers,
                    "path": path,
                    "method": method,
                    "summary": details.get("summary", details.get("operationId",path)),
                    "description": details.get("description", ""),
                    "headers": {},
                    "query_params": {},
                    "body": None,
                    "path_params": {},
                }

                # Extract parameters
                parameters = details.get("parameters", [])
                request_info['query_params'] = self.extract_query_parameters_from_swagger(parameters,global_security)
                request_info['headers'] = self.extract_headers_from_swagger(parameters,methods,global_security)
                request_info['path_params'] = self.extract_path_parameters_from_swagger(parameters,methods,path )
                request_info['auth'] = self.extract_security_details(global_security,details)
                request_info['collection_name'] = collection_name



                # Extract request body
                if "consumes" in details:

                    parameters = details.get("parameters",[])
                    resolved_body = self.resolve_recursive(parameters)
                    filtered_body = [each for each in resolved_body if each.get("in",'') in ["body"]]
                    filtered_form_body = [each for each in resolved_body if each.get("in",'') in ["formData"]]

                    if filtered_body:
                        for ele in filtered_body:
                                content = {}
                                for con in details['consumes']:
                                    content.update({con:{"schema":ele.get("schema",{})}})
                                ele.update({"content":content})

                        request_info["body"] = filtered_body

                    elif filtered_form_body:
                            content = {}
                            for con in details['consumes']:
                                content.update({con: {"schema": {"properties":{val.get("name"):val for val in filtered_form_body}}}})

                            # filtered_form_body.update({"content": content})

                            request_info["body"] =  [{"content": content}]


                elif any([p.get("in",'') in ["body", "formData"] for p in  details.get("parameters",[])]):
                    parameters = details.get("parameters", [])
                    resolved_body = self.resolve_recursive(parameters)
                    filtered_body = [each for each in resolved_body if each.get("in", '') in ["body", "formData"]]
                    for ele in filtered_body:
                        map_content = {"body":"application/json","formData":"application/x-www-form-urlencoded"}
                        content = {}
                        content.update({map_content.get(ele.get("in",'')): {"schema": ele.get("schema", {})}})
                        ele.update({"content": content})

                    request_info["body"] = filtered_body

                if "produces" in details:
                    accept_contents = []
                    for accept in details["produces"]:
                        accept_contents.append(accept)
                    request_info["request_content_types"] = accept_contents

                if "responses" in details:
                    try:
                        success_code =  [item for item in details["responses"] if  item.isdigit() and (200 <=  int(item) < 300)]
                        if len(success_code) > 0:
                            resolved_body = self.resolve_recursive(details["responses"][success_code[0]])
                            request_info["response"] = resolved_body
                        else:
                            request_info["response"] = {}
                    except ValueError as e :
                        if 'default' in details["responses"]:
                            resolved_body = self.resolve_recursive(details["responses"]['default']['content']['application/json'])
                            request_info["response"] = resolved_body
                    except Exception as e:
                        request_info["response"] = {}

                request_collection.append(request_info)

        return request_collection

    def save_collection(self, collection, output_file):
        """Saves the request collection to a JSON file."""
        def custom_serializer(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(output_file, 'w') as file:
            json.dump(collection, file, indent=4, default=custom_serializer)

    def get_body(self, schema):
        def extract_properties(schema):
            """Extract properties from schema, considering all possible locations."""
            properties = {}

            # If 'properties' is directly in schema
            if 'properties' in schema:
                properties.update(schema['properties'])

            # If 'items' contains 'properties'
            if 'items' in schema and 'properties' in schema['items']:
                properties.update(schema['items']['properties'])

            # If 'allOf' is in schema, merge properties from each element
            if 'allOf' in schema:
                for sub_schema in schema['allOf']:
                    properties.update(extract_properties(sub_schema))

            return properties

        def extract_required(schema):
            """Extract required fields from schema."""
            required = schema.get("required", [])

            # If 'allOf' is in schema, combine required fields from each element
            if 'allOf' in schema:
                for sub_schema in schema['allOf']:
                    required += extract_required(sub_schema)

            if 'items' in schema and 'required' in schema['items']:
                required +=  schema['items']['required']

            return list(set(required))  # Remove duplicates

        # Extract properties and required fields
        properties = extract_properties(schema)
        required = extract_required(schema)
        data_type = schema.get("type",{})

        # Rearrange properties: required first, then others
        rearranged = {key: properties[key] for key in required if key in properties}
        rearranged.update({key: properties[key] for key in properties if key not in required})

        final_body = {}

        for data, key in rearranged.items():

            if key.get("allOf"):
                body_of_allof = {}
                for sub_schema in key["allOf"]:

                    result = self.extract_key_types(data, {data: sub_schema})
                    newkey, value = next(iter(result.items()))

                    if not isinstance(value,dict):
                        body_of_allof = value
                    else:
                        body_of_allof.update(**value)

                    # body_of_allof.update(self.get_body(sub_schema))

                final_body[data] = body_of_allof
            elif key.get("properties"):
                result = self.extract_key_types(data, {data: key})
                key, value = next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = self.get_body(key)

            elif key.get("items"):
                result = self.extract_key_types(data, {data: key})
                key, value = next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = self.get_body(key)
            else:
                result = self.extract_key_types(data, {data:key})
                key, value =  next(iter(result.items()))
                final_body[key] = value

                # final_body[data] = key.get("example", f"{key.get('description','')} {key.get('format', '')} {key['type']}")


        return [final_body] if data_type =="array" else final_body



    def resolve_ref_url(self, ref, spec):
        """Resolves a $ref within the OpenAPI spec."""
        if not ref.startswith("#/"):
            raise ValueError(f"Unsupported $ref format: {ref}")

        path = ref.lstrip("#/").split("/")
        ref_data = spec
        for part in path:
            ref_data = ref_data.get(part, {})
        return ref_data

    def extract_and_replace_urls(self,urls,servers):
        final_urls = []
        for url,server in zip(urls,servers):
            region_info = server.get('variables', {}).get('region', {})
            region = region_info.get('default') or region_info.get('enum', [])[0] if len(region_info.get('enum', [])) > 0 else []
            if region:
                final_urls.append(url.replace('{region}',region))
        return final_urls

    def extract_server_urls(self,schemes,host,basePath):
        default_url = "http://localhost:0001"
        urls = []

        if host and basePath:
            for scheme in schemes:
                    url = scheme+"://"+host+basePath
                    urls.append(url)
        return urls if urls else [default_url]


    def dict_to_xml(self,tag, schema):
        elem = ET.Element(tag)

        for key, value in schema.items():
            if isinstance(value, dict):
                if "properties" in value:
                    # Nested object
                    child = self.dict_to_xml(value.get("xml", {}).get("name", key), value["properties"])
                    elem.append(child)
                elif "example" in value:
                    # Leaf node with an example
                    child = ET.Element(key)
                    child.text = str(value["example"])
                    elem.append(child)
                elif "items" in value:
                    # Handle array
                    array_tag = key
                    wrapped = value.get("xml", {}).get("wrapped", False)

                    if wrapped:
                        wrapper = ET.Element(array_tag)
                        for item in value["items"].get("xml", []):
                            if "properties" in value["items"]:
                                child = self.dict_to_xml(value["items"].get("xml", {}).get("name", key), value["items"]["properties"])
                                elem.append(child)
                            else:
                                item_tag = value["items"].get("xml", {}).get("name", "item")
                                item_elem = ET.Element(item_tag)
                                item_elem.text = str(item)
                                wrapper.append(item_elem)
                            elem.append(wrapper)
                    else:
                        for item in value["items"].get("example", []):
                            item_elem = ET.Element(array_tag)
                            item_elem.text = str(item)
                            elem.append(item_elem)

            elif isinstance(value, list):
                # Handle list of primitive values
                for item in value:
                    child = ET.Element(key)
                    child.text = str(item)
                    elem.append(child)

        return elem

    def generate_xml_body(self,spec: Dict[str, Any]) -> str:
        type_defaults = {
            "string": "string",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {}
        }

        def create_element(name: str, properties: dict, parent: ET.Element) -> None:
            # Handle wrapped arrays
            if properties.get('type') == 'array' and properties.get('xml', {}).get('wrapped'):
                wrapper = ET.SubElement(parent, name)
                item_props = properties['items']
                item_name = item_props.get('xml', {}).get('name', 'item')

                # Create example array item
                if 'properties' in item_props:
                    sub_elem = ET.SubElement(wrapper, item_name)
                    for prop_name, prop_value in item_props['properties'].items():
                        create_element(prop_name, prop_value, sub_elem)
                else:
                    sub_elem = ET.SubElement(wrapper, item_name)
                    sub_elem.text = item_props.get('example', 'string')
                return

            # Handle regular properties
            element = ET.SubElement(parent, name)
            if properties.get('type') == 'object' and 'properties' in properties:
                for prop_name, prop_value in properties['properties'].items():
                    create_element(prop_name, prop_value, element)
            else:
                if 'example' in properties:
                    element.text = str(properties['example'])
                elif 'enum' in properties:
                    element.text = str(properties['enum'][0])
                elif 'type' in properties:
                    element.text = str(type_defaults[properties['type']])
                else:
                    element.text = ''

        # Create root element
        root_name = spec.get('xml', {}).get('name', 'root')
        root = ET.Element(root_name)

        # Process all properties
        for prop_name, prop_value in spec.get('properties',{}).items():
            create_element(prop_name, prop_value, root)

        # Convert to string with pretty printing
        if not list(root):
           xml_str ='''<?xml version="1.0" encoding="UTF-8"?>
<!-- XML example cannot be generated; root element name is undefined -->'''
        else:
         xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml()

        # Add XML declaration
        if not xml_str.startswith('<?xml'):
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

        return xml_str



    def convert_to_form_data(self,data):

        form_data = []

        if isinstance(data,list):
            data = data[0]

        for key, value in data.items():

            form_data.append({
                "key": key,
                "value": json.dumps(value) if isinstance(value, dict) or isinstance(value, list) else str(value),
                "active": True
            })

        return form_data

    def convert_into_prutan_spec(self,data):

        specs = []

        for request in data:
            servers = request['servers']
            if not servers:
                print("No server url's present")
                return specs

            base_spec = {
                'collection_name':request['collection_name'],
                'method': request['method'].upper(),
                'endpoint': servers[0] + request["path"],
                'request_name':request["summary"],
                'response_content': request.get("request_content_types", [])
            }
            path_params = request['path_params'] if request['path_params']  else []

            if request.get('response'):
                for content in request['response'].get('content', []):
                    if content == "application/json":
                        if isinstance(request['response'].get('content', {}).get(content, {}), dict):
                            response = self.get_body(request['response'].get('content', {}).get(content, {})['schema'])
                        elif isinstance(request['response'].get('content', {}).get(content, {}), set):
                            response = str(request['response'].get('content', {}).get(content))
                        else:
                            response = request['response'].get('content', {}).get(content)
                        base_spec["responseContent"] = json.dumps(response) if isinstance(response,(dict, list)) else str(response)
                        break

            # Handle body content if it exists

            if base_spec["response_content"] == []:
                base_spec["response_content"] = ['application/json']

            if request.get('body'):
                for bod in request['body']:# Ideally the length of element is 1
                    for content in bod.get('content', []):
                        spec = base_spec.copy()
                        for accept in spec['response_content']:
                            header_copy = request.get('headers', []).copy()
                            spec["content_type"] = content
                            if content == "application/json":
                                if isinstance(bod.get('content', {}).get(content, {}), dict):
                                    spec["body"] = self.get_body(
                                        bod.get('content', {}).get(content, {})['schema'])
                                elif isinstance(bod.get('content', {}).get(content, {}), set):
                                    spec["body"] = str(bod.get('content', {}).get(content))
                                else:
                                    spec["body"] = bod.get('content', {}).get(content)
                            elif content == "application/xml":
                                body_contents = bod.get('content',{}).get(content,{})['schema']
                                xml_request_body = self.generate_xml_body(body_contents)
                                spec["body"] = xml_request_body
                            elif content == "application/x-www-form-urlencoded":
                                form_body = self.get_body(bod.get('content',{}).get(content,{})['schema'])
                                spec['body'] = self.convert_to_form_data(form_body)

                            elif  content == "multipart/form-data":
                                form_body = self.get_body(bod.get('content', {}).get(content, {})['schema'])
                                spec['body'] = self.convert_to_form_data(form_body)

                            elif content == "application/octet-stream":
                                form_body = self.get_body(bod.get('content', {}).get(content, {})['schema'])
                                spec['body'] = form_body

                            header_copy.append({
                                "name": "accept",
                                "value": accept,
                                "enabled": True
                            })

                            # if request.get('headers'):
                            #     spec["headers"] = request['headers']

                            if header_copy:
                                spec["headers"] = header_copy

                            if request.get('query_params'):
                                spec["params"] = request['query_params']
                                url = base_spec['endpoint']
                                for param in request['query_params']:
                                    url += f"?{param.get("name")}={param.get("value", '')}"
                                base_spec['endpoint'] = url
                            if request.get('path_params'):
                                spec["path_params"] = request['path_params']
                            if request.get('auth'):
                                spec["auth"] = request['auth']

                            spec_copy = spec.copy()
                            specs.append(spec_copy)
            else:
                # if request.get('headers'):
                #     base_spec["headers"] = request['headers']
                for accept in base_spec["response_content"]:
                    header_copy = request.get('headers', []).copy()
                    # header_copy.append({"accept": accept})
                    header_copy.append({
                        "name": "accept",
                        "value": accept,
                        "enabled": True
                    })
                    if header_copy:
                        base_spec["headers"] = header_copy

                    if request.get('query_params'):
                        base_spec["params"] = request['query_params']
                        # updating query params in url
                        url = base_spec['endpoint']
                        for param in request['query_params']:
                            url+=f"?{param.get("name")}={param.get("value",'')}"
                        base_spec['endpoint']  = url


                    if request.get('path_params'):
                        base_spec["path_params"] = request['path_params']

                    spec_copy = base_spec.copy()
                    specs.append(spec_copy)

        return specs

    def process_schema(self,schema,prop=''):
        """
        Process the schema and return its value based on the type and other conditions.
        """

        if not isinstance(schema,dict):
            return schema
        if "example" in schema:
            if isinstance(schema['example'],datetime) or isinstance(schema['example'],date):
                return str(schema["example"])
            return schema["example"]

        if "examples" in schema:
            # If examples are provided, return an empty list of the correct type
            return []

        if "enum" in schema:
            value = schema["enum"][0] if schema["enum"] else None
            return value

        if "nullable" in schema and schema["nullable"]:
            return None

        if schema.get("type") == "object":
            result = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                result[prop] = self.process_schema(prop_schema,prop)
            if "additionalProperties" in schema:
                additional_schema = schema["additionalProperties"]
                for i in range(3):
                   result[f"additionalProp{i+1}"] = self.process_schema(additional_schema)
            return result

        if schema.get("type") == "array":
            items_schema = schema.get("items", {})
            return [self.process_schema(items_schema)]

        if "allOf" in schema:
            result = {}
            for sub_schema in schema["allOf"]:
                val = self.process_schema(sub_schema)
                if not isinstance(val,dict):
                    result = val
                else:
                   result.update(val)
            return result

        if "oneOf" in schema or "anyOf" in schema:
            options = schema.get("oneOf") or schema.get("anyOf")
            return [self.process_schema(option) for option in options]

        if "discriminator" in schema:
            # Handle discriminator if necessary (usually requires additional logic)
            return {"discriminator": schema["discriminator"]}

        if 'properties' in schema:
            v = self.get_body(schema)
            return v

        # Return a default value based on the type
        type_defaults = {
            "string": "string",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {}
        }
        return type_defaults.get(schema.get("type"), type_defaults.get("string"))


    def extract_key_types(self,key, value):
        """
        Extract key types and their values from a schema.
        """
        result = {}

        value = value[key]

        if value.get("type") == "array":
            item_type = self.process_schema(value.get("items", {}))
            result[key] = [item_type]
        elif value.get("type") == "object":
            result[key] = self.process_schema(value)
        elif "enum" in value:
            result[key] = value["enum"][0] if value["enum"] else None
        elif 'properties' in value:
             v = self.get_body(value)
             result[key] = v
        else:
            result[key] = self.process_schema(value)

        return result