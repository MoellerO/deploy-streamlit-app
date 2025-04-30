import streamlit as st
from utils.auth import Auth
from config_file import Config
import streamlit as st
import uuid
import asyncio
import boto3
from typing import Optional, Dict, Any, List, Tuple


class BedrockAgentChat:
    def __init__(self, agent_id: str, agent_alias_id: str, region_name: Optional[str] = "us-west-2"):
        """
        Initialize the Bedrock Agent Chat interface.

        Args:
            agent_id: The ID of your Bedrock Agent
            agent_alias_id: The alias ID of your Bedrock Agent
            region_name: AWS region (optional, uses default if not specified)
        """
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        # Initialize the Bedrock Agent Runtime client
        self.bedrock_agent_runtime_client = boto3.client(
            'bedrock-agent-runtime',
            region_name=region_name
        )

    async def invoke_agent(self, prompt: str, session_id: str, thinking_placeholder) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Invoke the Bedrock Agent with a prompt and handle both text and files.

        Args:
            prompt: User's input text
            session_id: Unique session identifier
            thinking_placeholder: Streamlit placeholder to update with thinking process

        Returns:
            Tuple containing:
            - The agent's text response
            - List of file objects (images, etc.)
        """
        try:
            thinking_placeholder.write(
                "🔍 Sending your request to the agent...")

            response = self.bedrock_agent_runtime_client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
                enableTrace=True,  # Enable tracing for better debugging
            )

            completion = ""
            files = []
            thinking_steps = []
            current_step = ""

            thinking_placeholder.write("🧠 Agent is processing your request...")
            file_names_seen = set()

            # Process the event stream to collect both text and files
            for event_idx, event in enumerate(response.get("completion", [])):
                # Process trace events to show thinking steps
                if 'trace' in event:
                    trace_data = event['trace']

                    print(trace_data)

                    # Extract thinking steps from trace
                    if 'thinking' in trace_data:
                        thinking_step = trace_data['thinking']
                        if thinking_step and thinking_step != current_step:
                            current_step = thinking_step
                            thinking_steps.append(thinking_step)
                            # Update the thinking placeholder with the latest step
                            thinking_content = "\n\n".join(
                                [f"🤔 {step}" for step in thinking_steps])
                            thinking_placeholder.markdown(
                                f"### Agent's thinking process:\n{thinking_content}")

                    # Show when agent is accessing knowledge bases
                    if 'knowledgeBaseLookup' in trace_data:
                        kb_info = trace_data['knowledgeBaseLookup']
                        kb_step = f"Searching knowledge base: {kb_info.get('knowledgeBaseId', 'Unknown KB')}"
                        thinking_steps.append(kb_step)
                        thinking_content = "\n\n".join(
                            [f"🤔 {step}" for step in thinking_steps])
                        thinking_placeholder.markdown(
                            f"### Agent's thinking process:\n{thinking_content}")

                    # Show when agent is using action groups
                    if 'actionGroupInvocation' in trace_data:
                        action_info = trace_data['actionGroupInvocation']
                        action_step = f"Invoking action: {action_info.get('actionGroup', 'Unknown action')} - {action_info.get('apiPath', '')}"
                        thinking_steps.append(action_step)
                        thinking_content = "\n\n".join(
                            [f"🤔 {step}" for step in thinking_steps])
                        thinking_placeholder.markdown(
                            f"### Agent's thinking process:\n{thinking_content}")
                        thinking_placeholder.write("Invoking Action...")

                # Process text chunks
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        completion += chunk['bytes'].decode('utf-8')
                        thinking_placeholder.write("📝 Composing response...")

                # Process file outputs (visualizations, etc.)
                if 'files' in event:
                    event_files = event['files'].get('files', [])
                    if event_files:
                        thinking_placeholder.write(
                            f"🖼️ Generating visualizations ({len(event_files)} files)...")

                    for file_idx, file in enumerate(event_files):
                        file_name = file.get(
                            'name', f"visualization_{len(files)}.png")
                        # Skip if we've already seen this file name
                        if file_name in file_names_seen:
                            continue
                        file_names_seen.add(file_name)

                        file_data = {
                            'name': file.get('name', f"visualization_{len(files)}.png"),
                            'type': file.get('type', 'image/png'),
                            'bytes': file.get('bytes', b'')
                        }

                        files.append(file_data)

            thinking_placeholder.write("✅ Response complete!")

            return completion, files

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            thinking_placeholder.error(f"❌ {error_msg}")
            return f"Error: {str(e)}", []


st.set_page_config(
    page_title="Endrich x AWS: IoT Visualizer",
    page_icon="🤖",
    layout="wide",  # Changed to wide layout for better visualization display
    initial_sidebar_state="expanded"
)

# ID of Secrets Manager containing cognito parameters
secrets_manager_id = Config.SECRETS_MANAGER_ID
# ID of the AWS region in which Secrets Manager is deployed
region = Config.DEPLOYMENT_REGION

# Initialise CognitoAuthenticator
authenticator = Auth.get_authenticator(secrets_manager_id, region)

# Authenticate user, and stop here if not logged in
is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()


def logout():
    authenticator.logout()


with st.sidebar:
    st.text(f"Welcome,\n{authenticator.get_username()}")
    st.button("Logout", "logout_btn", on_click=logout)

st.title("Endrich x AWS: IoT Visualizer")


def main():
    # Initialize session state for chat history, session ID, and visualizations
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "visualizations" not in st.session_state:
        st.session_state.visualizations = []

    # Initialize agent chat interface
    agent_chat = BedrockAgentChat(
        agent_id=Config.AGENT_ID,
        agent_alias_id=Config.AGENT_ALIAS_ID,
        region_name=Config.AGENT_REGION
    )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Display any visualizations attached to this message
            if "files" in message and message["files"]:
                for file_idx, file in enumerate(message["files"]):
                    if file["type"].startswith("image/"):
                        try:
                            st.image(file["bytes"], caption=file["name"])
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    else:
                        # For non-image files, offer a download button
                        st.download_button(
                            label=f"Download {file['name']}",
                            data=file["bytes"],
                            file_name=file["name"],
                            mime=file["type"]
                        )

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt, "files": []}
        )

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response with a spinner while waiting
        with st.chat_message("assistant"):
            # Create a placeholder for the thinking process
            thinking_placeholder = st.empty()

            # Run the async function to get the agent's response and any files
            response_text, response_files = asyncio.run(agent_chat.invoke_agent(
                prompt=prompt,
                session_id=st.session_state.session_id,
                thinking_placeholder=thinking_placeholder
            ))

            # Clear the thinking placeholder after processing is complete
            thinking_placeholder.empty()

            # Display the text response
            st.write(response_text)

            # Display any visualizations
            for file in response_files:
                if file["type"].startswith("image/"):
                    try:
                        st.image(file["bytes"], caption=file["name"])
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                else:
                    # For non-image files, offer a download button
                    st.download_button(
                        label=f"Download {file['name']}",
                        data=file["bytes"],
                        file_name=file["name"],
                        mime=file["type"]
                    )

            # Store files for visualization gallery
            if response_files:
                st.session_state.visualizations.extend(response_files)

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "files": response_files
            })

    # Add buttons to the sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("New Conversation"):
        st.session_state.chat_history = []
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.visualizations = []
        if "debug_events" in st.session_state:
            del st.session_state.debug_events
        if "debug_log" in st.session_state:
            del st.session_state.debug_log
        st.rerun()

    if st.sidebar.button("Clear Visualization Gallery"):
        st.session_state.visualizations = []
        st.rerun()

    # Add visualization gallery section
    if st.session_state.visualizations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Visualization Gallery")
        for idx, file in enumerate(st.session_state.visualizations):
            if file["type"].startswith("image/"):
                st.sidebar.image(
                    file["bytes"], caption=file["name"], width=200)


if __name__ == "__main__":
    main()
