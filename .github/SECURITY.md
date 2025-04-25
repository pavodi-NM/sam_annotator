# SAM Annotator Security Policy

## Reporting a Vulnerability

### Preferred Reporting Method

SAM Annotator utilizes GitHub's private vulnerability reporting feature. To report a security vulnerability:

1. Go to the [SAM Annotator repository](https://github.com/pavodi-nm/sam_annotator)
2. Navigate to the "Security" tab
3. Select "Report a vulnerability"
4. Fill out the form with all required details

This method ensures that your report remains confidential and is only visible to repository maintainers.

### What Information to Include

Please include the following information in your vulnerability report:

- **Vulnerability Description**: A clear and concise explanation of the security issue
- **Affected Component(s)**: Which part of SAM Annotator contains the vulnerability
- **Reproduction Steps**: Detailed steps to reproduce the vulnerability
- **Potential Impact**: Your assessment of the severity and potential consequences
- **Environment Details**: Relevant information about the environment where the vulnerability was discovered
  - Operating system
  - Python version
  - SAM Annotator version
  - Relevant dependencies and their versions
- **Suggested Fix** (optional): If you have insights into how the vulnerability might be addressed
- **Your Contact Information**: How we can reach you for follow-up questions

### Scope

This security policy covers:

- The SAM Annotator application code
- Dependencies explicitly required by SAM Annotator
- Documentation that might lead to insecure usage

Out of scope:

- Issues in third-party dependencies that have not been publicly disclosed yet (please report these to the respective projects first)
- Issues that require physical access to a user's device
- Theoretical vulnerabilities without practical demonstration
- Issues in development or test environments

### Response Timeline

We are committed to addressing security vulnerabilities promptly:

- **Acknowledgment**: You will receive an acknowledgment of your report within 48 hours
- **Validation**: We aim to validate reports within 1 week
- **Regular Updates**: For valid reports, you will receive updates at least once a week
- **Resolution Timeline**: Critical vulnerabilities will be prioritized with a target resolution time of 30 days or less

### Disclosure Policy

- We follow a coordinated disclosure process
- We request that reporters maintain confidentiality until a fix is released
- After a fix is available, we will work with the reporter on an appropriate disclosure timeline
- Reporters will be credited in the security advisory unless they request to remain anonymous

### Recognition

We believe in acknowledging security researchers who help improve our project's security. With your permission, we will:

- Credit you in our security advisories
- Add your name to a security acknowledgments section in our documentation
- Potentially include you in our GitHub Security Advisories

Thank you for helping make SAM Annotator secure for everyone! 